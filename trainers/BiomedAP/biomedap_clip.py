"""
BiomedAP (CLIP backbone)
========================
Adaptive Prompt Learning with Cross-Modal Fusion

核心创新:
1. Visual Prompt：在图像编码器的多层注入可学习的prompt tokens
2. Text Prompt：可学习的上下文向量
3. 跨模态融合：在指定层进行 Visual-Text Prompt 交互
4. 多质量文本蒸馏：高质量（50模板）+ 低质量（单模板）
5. 四重损失优化：CE + L1_high + KL + L1_low + Alignment

损失函数:
L = L_ce + λ1*L_L1_high + λ2*L_KL + λ3*L_L1_low + λ4*L_alignment
"""

# ========== 环境配置 ==========
import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import copy
import os.path as osp
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import CUSTOM_BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import ZERO_SHOT_TEMPLATES
from trainers.cross_modal_fusion import CrossModalPromptFusion, LightweightCrossModalFusion

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    """加载 CLIP 模型到 CPU"""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Prompt 学习器（集成高质量 + 低质量特征预计算）"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDAP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化可学习上下文向量
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # 处理类名
        classnames = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        # 使用中等质量模板
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        # 加载教师模型
        clip_model_temp = load_clip_to_cpu(cfg).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg).float().cuda()
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            
            # ========== 预计算高质量特征（50个模板）==========
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDAP.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([
                    clip.tokenize(BIOMEDDPT_TEMPLATES[classname][i]) 
                    for classname in classnames
                ])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)  # [N, 50, 512]
        
        # ========== 预计算低质量特征（鲁棒性锚点）==========
        print("[ANCHOR] Loading low-quality Prompt (robustness anchor)")
        low_template_type = cfg.TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE
        
        if low_template_type not in ZERO_SHOT_TEMPLATES:
            print(f"Warning: Unknown template type '{low_template_type}', using 'minimal'")
            low_template_type = "minimal"
        
        template = ZERO_SHOT_TEMPLATES[low_template_type]
        print(f"Low-quality template type: {low_template_type}")
        
        # 生成低质量 Prompt
        if template == "":
            low_quality_prompts = ["X" for _ in classnames]
            print("Using 'X' as low-quality prompt")
        else:
            low_quality_prompts = [
                template.format(**{"class": cls}) for cls in classnames
            ]
            print(f"Low-quality prompt examples (first 3):")
            for cls, prompt in zip(classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        # 预计算低质量特征
        with torch.no_grad():
            low_tokenized = torch.cat([
                clip.tokenize(p if p else "X") for p in low_quality_prompts
            ])
            low_text_features = clip_model_temp.encode_text(low_tokenized.cuda())
        
        self.fixed_low_embeddings = low_text_features  # [N, 512]
        print(f"[OK] Low-quality Prompt initialized\n")
        
        del clip_model_temp
        del clip_model_temp_image

        # 保存 token 前缀和后缀
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """构造完整的 Prompt"""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        """前向传播"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CLIP_Inplanted(nn.Module):
    """带Visual Prompt和跨模态交互的图像编码器"""
    def __init__(self, clip_model, enable_fusion=False, fusion_layers=[5, 8]):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.num_tokens = 4
        self.prompt_dim = 768
        self.prompt_embeddings = torch.zeros(1, self.num_tokens, self.prompt_dim)
        self.deep_prompt_embeddings = torch.zeros(12, self.num_tokens, self.prompt_dim)
        self.prompt_dropout = nn.Dropout(0.5)

        # ========== 【修改】动态获取text prompt的维度 ==========
        # CLIP的text embedding维度通常是512，不是768
        text_dim = clip_model.ln_final.weight.shape[0]  # 获取实际的text维度
        print(f"[FUSION] Detected text_dim={text_dim}, visual_dim={self.prompt_dim}")
        
        # ========== 跨模态融合配置 ==========
        self.enable_fusion = enable_fusion
        self.fusion_layers = fusion_layers
        
        if enable_fusion:
            print(f"[FUSION] Enabling Cross-Modal Fusion at layers: {fusion_layers}")
            self.fusion_modules = nn.ModuleDict({
                str(layer): CrossModalPromptFusion(
                    visual_dim=self.prompt_dim, 
                    text_dim=text_dim, 
                    num_heads=8, 
                    dropout=0.1)
                for layer in fusion_layers
            })
            print(f"[FUSION] Created {len(self.fusion_modules)} fusion modules")
        
        self.visual_prompts_cache = {}

    def forward(self, x, text_prompts=None, label=None):
        """
        注入Visual Prompt并与Text Prompt交互的前向传播
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            text_prompts: Text Prompt特征 [n_cls, n_ctx, 512] (可选)
            label: 真实标签 [B] (训练时使用)
        """
        # Patch embedding
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.image_encoder.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        
        # 注入浅层Visual Prompt
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1).cuda()),
            x[:, 1+self.num_tokens:, :]
        ), dim=1)

        # ========== 根据训练/测试选择不同策略 ==========
        if self.training and label is not None:
            # ========== 训练时：标准流程 ==========
            for i in range(12):
                B = x.shape[0]
                
                # 注入深层Visual Prompt
                current_visual_prompts = self.prompt_dropout(
                    self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()
                )
                x = torch.cat((
                    x[:, :1, :],
                    current_visual_prompts,
                    x[:, 1+self.num_tokens:, :]
                ), dim=1)
                
                # 在指定层融合
                if self.enable_fusion and i in self.fusion_layers and text_prompts is not None:
                    curr_visual_prompts = x[:, 1:1+self.num_tokens, :]
                    curr_text_prompts = text_prompts[label]
                    
                    fusion_module = self.fusion_modules[str(i)]
                    enhanced_visual, enhanced_text = fusion_module(
                        curr_visual_prompts,
                        curr_text_prompts
                    )
                    
                    x = torch.cat([
                        x[:, :1, :],
                        enhanced_visual,
                        x[:, 1+self.num_tokens:, :]
                    ], dim=1)
                    
                    self.visual_prompts_cache[f'layer_{i}_text'] = enhanced_text
                
                # 执行Transformer block
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.image_encoder.transformer.resblocks[i](x)
                x = x.permute(1, 0, 2)  # LND -> NLD
            
            # 最终投影
            x = self.image_encoder.ln_post(x[:, 0, :])
            if self.image_encoder.proj is not None:
                x = x @ self.image_encoder.proj
            return x

        else:
            # ========== 【修改】测试时：使用平均text prompts ==========
            for i in range(12):
                B = x.shape[0]
                
                # 注入深层Visual Prompt
                current_visual_prompts = self.prompt_dropout(
                    self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()
                )
                x = torch.cat((
                    x[:, :1, :],
                    current_visual_prompts,
                    x[:, 1+self.num_tokens:, :]
                ), dim=1)
                
                # ========== 【关键改进】测试时使用所有类别的平均text prompts ==========
                if self.enable_fusion and i in self.fusion_layers and text_prompts is not None:
                    curr_visual_prompts = x[:, 1:1+self.num_tokens, :]
                    # 使用所有类别的平均text prompts
                    curr_text_prompts = text_prompts.mean(dim=0).unsqueeze(0).expand(B, -1, -1)
                    
                    fusion_module = self.fusion_modules[str(i)]
                    enhanced_visual, enhanced_text = fusion_module(
                        curr_visual_prompts,
                        curr_text_prompts
                    )
                    
                    x = torch.cat([
                        x[:, :1, :],
                        enhanced_visual,
                        x[:, 1+self.num_tokens:, :]
                    ], dim=1)
                    
                    self.visual_prompts_cache[f'layer_{i}_text'] = enhanced_text
                
                # 执行Transformer block
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.image_encoder.transformer.resblocks[i](x)
                x = x.permute(1, 0, 2)  # LND -> NLD
            
            # 最终投影
            x = self.image_encoder.ln_post(x[:, 0, :])
            if self.image_encoder.proj is not None:
                x = x @ self.image_encoder.proj
            return x
        
        # # ========== 根据训练/测试选择不同策略 ==========
        # if self.training and label is not None:
        #     # ========== 训练时：标准流程 ==========
        #     for i in range(12):
        #         B = x.shape[0]
                
        #         # 注入深层Visual Prompt
        #         current_visual_prompts = self.prompt_dropout(
        #             self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()
        #         )
        #         x = torch.cat((
        #             x[:, :1, :],
        #             current_visual_prompts,
        #             x[:, 1+self.num_tokens:, :]
        #         ), dim=1)
                
        #         # 在指定层融合
        #         if self.enable_fusion and i in self.fusion_layers and text_prompts is not None:
        #             curr_visual_prompts = x[:, 1:1+self.num_tokens, :]
        #             curr_text_prompts = text_prompts[label]
                    
        #             fusion_module = self.fusion_modules[str(i)]
        #             enhanced_visual, enhanced_text = fusion_module(
        #                 curr_visual_prompts,
        #                 curr_text_prompts
        #             )
                    
        #             x = torch.cat([
        #                 x[:, :1, :],
        #                 enhanced_visual,
        #                 x[:, 1+self.num_tokens:, :]
        #             ], dim=1)
                    
        #             self.visual_prompts_cache[f'layer_{i}_text'] = enhanced_text
                
        #         # 执行Transformer block
        #         x = x.permute(1, 0, 2)  # NLD -> LND
        #         x = self.image_encoder.transformer.resblocks[i](x)
        #         x = x.permute(1, 0, 2)  # LND -> NLD
            
        #     # 最终投影
        #     x = self.image_encoder.ln_post(x[:, 0, :])
        #     if self.image_encoder.proj is not None:
        #         x = x @ self.image_encoder.proj
        #     return x
        
        # else:
        #     # ========== 测试时：不同策略 ==========
        #     if not self.enable_fusion or text_prompts is None:
        #         # 如果未启用融合，直接走标准流程
        #         for i in range(12):
        #             B = x.shape[0]
        #             current_visual_prompts = self.prompt_dropout(
        #                 self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()
        #             )
        #             x = torch.cat((
        #                 x[:, :1, :],
        #                 current_visual_prompts,
        #                 x[:, 1+self.num_tokens:, :]
        #             ), dim=1)
        #             x = x.permute(1, 0, 2)
        #             x = self.image_encoder.transformer.resblocks[i](x)
        #             x = x.permute(1, 0, 2)
                
        #         x = self.image_encoder.ln_post(x[:, 0, :])
        #         if self.image_encoder.proj is not None:
        #             x = x @ self.image_encoder.proj
        #         return x
            
        #     # ========== 测试时：为每个类别分别提取特征 ==========
        #     assert self.enable_fusion, "[ERROR] Fusion should be enabled in test mode!"
        #     assert text_prompts is not None, "[ERROR] text_prompts is None!"
        #     print(f"[INFO] Test mode: Multi-class fusion activated, n_cls={text_prompts.shape[0]}")
            
        #     n_cls = text_prompts.shape[0]
        #     all_image_features = []
            
        #     for c in range(n_cls):
        #         # 为第c个类别提取特征
        #         x_c = x.clone()
                
        #         for i in range(12):
        #             current_visual_prompts = self.prompt_dropout(
        #                 self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()
        #             )
        #             x_c = torch.cat((
        #                 x_c[:, :1, :],
        #                 current_visual_prompts,
        #                 x_c[:, 1+self.num_tokens:, :]
        #             ), dim=1)
                    
        #             # 在指定层融合（使用第c个类别的text prompt）
        #             if i in self.fusion_layers:
        #                 curr_visual_prompts = x_c[:, 1:1+self.num_tokens, :]
        #                 curr_text_prompts = text_prompts[c:c+1].expand(B, -1, -1)
                        
        #                 fusion_module = self.fusion_modules[str(i)]
        #                 enhanced_visual, _ = fusion_module(
        #                     curr_visual_prompts,
        #                     curr_text_prompts
        #                 )
                        
        #                 x_c = torch.cat([
        #                     x_c[:, :1, :],
        #                     enhanced_visual,
        #                     x_c[:, 1+self.num_tokens:, :]
        #                 ], dim=1)
                    
        #             # 执行Transformer block
        #             x_c = x_c.permute(1, 0, 2)
        #             x_c = self.image_encoder.transformer.resblocks[i](x_c)
        #             x_c = x_c.permute(1, 0, 2)
                
        #         # 最终投影
        #         x_c = self.image_encoder.ln_post(x_c[:, 0, :])
        #         if self.image_encoder.proj is not None:
        #             x_c = x_c @ self.image_encoder.proj
                
        #         all_image_features.append(x_c.unsqueeze(1))  # [B, 1, dim]
            
        #     # 堆叠所有类别的特征
        #     all_image_features = torch.cat(all_image_features, dim=1)  # [B, n_cls, dim]
            
        #     # 缓存供CustomCLIP使用
        #     print(f"[DEBUG] Writing to cache: shape={all_image_features.shape}")
        #     self.visual_prompts_cache['all_cls_features'] = all_image_features
            
        #     # 返回第一个类别的特征（兼容原接口）
        #     return all_image_features[:, 0, :]


class CustomCLIP(nn.Module):
    """自定义 CLIP 模型（集成跨模态融合 + 5个损失函数）"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.cfg = cfg
        
        # ========== 传入融合配置 ==========
        enable_fusion = cfg.TRAINER.BIOMEDAP.ENABLE_FUSION if hasattr(cfg.TRAINER.BIOMEDAP, 'ENABLE_FUSION') else False
        fusion_layers = cfg.TRAINER.BIOMEDAP.FUSION_LAYERS if hasattr(cfg.TRAINER.BIOMEDAP, 'FUSION_LAYERS') else [5, 8]
        
        self.image_encoder = CLIP_Inplanted(
            clip_model,
            enable_fusion=enable_fusion,
            fusion_layers=fusion_layers
        )
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # ========== Prompt对齐损失权重 ==========
        self.alignment_lambda = cfg.TRAINER.BIOMEDAP.ALIGNMENT_LAMBDA if hasattr(cfg.TRAINER.BIOMEDAP, 'ALIGNMENT_LAMBDA') else 0.0

    def forward(self, image, label=None):
        """前向传播(添加跨模态交互)"""
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # ========== 获取text prompts的中间表示 ==========
        prompts = self.prompt_learner()  # [n_cls, seq_len, 512]
        
        # 提取text context部分(去掉prefix和suffix)
        n_ctx = self.prompt_learner.n_ctx
        text_ctx = prompts[:, 1:1+n_ctx, :]  # [n_cls, n_ctx, 512]
        
        # ========== 提取图像特征(传入text prompts) ==========
        image_features = self.image_encoder(
            image.type(self.dtype),
            text_prompts=text_ctx,
            label=label
        )
        
        # 提取文本特征
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 高质量特征(教师)
        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 低质量特征(鲁棒性锚点)
        fixed_low_embeddings = self.prompt_learner.fixed_low_embeddings
        fixed_low_embeddings = fixed_low_embeddings / fixed_low_embeddings.norm(dim=-1, keepdim=True)
        
        if self.prompt_learner.training:
            # ========== 训练时：标准流程 ==========
            zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
            logits = logit_scale * image_features @ text_features.t()
            
            loss_ce = F.cross_entropy(logits, label)
            loss_l1_high = F.l1_loss(text_features, fixed_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_HIGH
            loss_kl = F.kl_div(F.log_softmax(logits, dim=1), F.log_softmax(zero_shot_logits, dim=1), reduction='sum', log_target=True) / logits.numel() * self.cfg.TRAINER.BIOMEDAP.KL_LAMBDA
            loss_l1_low = F.l1_loss(text_features, fixed_low_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_LOW
            
            loss_alignment = 0.0
            if self.alignment_lambda > 0 and hasattr(self.image_encoder, 'visual_prompts_cache'):
                cache = self.image_encoder.visual_prompts_cache
                if len(cache) > 0:
                    for key, enhanced_text in cache.items():
                        if 'layer' in key:
                            curr_text = text_ctx[label]
                            enhanced_pooled = F.normalize(enhanced_text.mean(dim=1), dim=-1)
                            curr_pooled = F.normalize(curr_text.mean(dim=1), dim=-1)
                            alignment_sim = (enhanced_pooled * curr_pooled).sum(dim=-1).mean()
                            loss_alignment += (1 - alignment_sim)
                    loss_alignment /= len([k for k in cache.keys() if 'layer' in k])
                    loss_alignment *= self.alignment_lambda
            
            total_loss = loss_ce + loss_l1_high + loss_kl + loss_l1_low + loss_alignment
            return logits, total_loss
        
        else:
            # ========== 【修改】测试时：标准logits计算 ==========
            logits = logit_scale * image_features @ text_features.t()
            return logits
        
            # # ========== 测试时：使用多类别特征 ==========
            # if hasattr(self.image_encoder, 'visual_prompts_cache') and 'all_cls_features' in self.image_encoder.visual_prompts_cache:
            #     print("[INFO] Using multi-class features from cache")
            #     all_image_features = self.image_encoder.visual_prompts_cache['all_cls_features']
            #     all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
            #     print(f"[INFO] Using multi-class features, shape: {all_image_features.shape}")
                
            #     logits = logit_scale * (all_image_features * text_features.unsqueeze(0)).sum(dim=-1)
            # else:
            #     print("[WARNING] all_cls_features not in cache, using fallback!")
            #     logits = logit_scale * image_features @ text_features.t()
            
            # return logits


@TRAINER_REGISTRY.register()
class BiomedAP_CLIP(TrainerX):
    """训练器"""
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.BIOMEDAP.PREC == "fp32" or cfg.TRAINER.BIOMEDAP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP with Cross-Modal Fusion")
        self.model = CustomCLIP(cfg, classnames, clip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        
        # ========== 添加fusion_modules到可训练参数 ==========
        names_to_update = ["prompt_learner.ctx"]
        
        if hasattr(cfg.TRAINER.BIOMEDAP, 'ENABLE_FUSION') and cfg.TRAINER.BIOMEDAP.ENABLE_FUSION:
            for name, param in self.model.named_parameters():
                if "fusion_modules" in name:
                    names_to_update.append(name)
                    param.requires_grad_(True)
        
        for name, param in self.model.named_parameters():
            if not any(update_name in name for update_name in names_to_update):
                param.requires_grad_(False)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        print(f"Parameters count: {len(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDAP.PREC == "amp" else None
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """前向和反向传播"""
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMEDAP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss = model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        """解析训练批次"""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        """加载模型"""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                print(f"No pretrained model found at '{model_path}', training from scratch")
                return

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} from {} (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
