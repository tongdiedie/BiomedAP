"""
BIOMEDAP (BiomedCLIP backbone)
==============================================
BiomedDPT + Text-Guided Visual Prompt + Multi-Quality Robustness

核心改进:
在 L1 损失中添加低质量 Prompt 约束，让模型同时学习：
1. 细粒度语义（从高质量 Prompt）
2. 核心语义（从低质量 Prompt）

损失函数:
L = L_ce + λ1 * L_L1_high + λ2 * L_KL + λ3 * L_L1_low
"""

# 【关键修复】禁用可能导致 sm80 错误的优化
import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from collections import OrderedDict
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import CUSTOM_BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import ZERO_SHOT_TEMPLATES  # 【新增】导入低质量模板
from trainers.cross_modal_fusion import CrossModalPromptFusion, LightweightCrossModalFusion

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        x = self.model.encode_text(prompts, True, tokenized_prompts)
        return x

class PromptLearner(nn.Module):
    """Prompt 学习器（添加了低质量 Prompt 约束）"""
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDAP.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        # 初始化 tokenizer
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化可学习的上下文向量
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.BIOMEDAP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # 处理类名
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        
        # 使用中等质量模板
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])
        
        # 加载教师模型（高质量 Prompt）
        biomedclip_model_temp, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            
            # 预计算高质量特征
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDAP.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i]) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)  # 高质量特征
        
        # ========== 【关键新增】预计算低质量特征 ==========
        print("[ANCHOR] Loading low-quality Prompt (robustness anchor)")
        low_template_type = cfg.TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE
        
        if low_template_type not in ZERO_SHOT_TEMPLATES:
            print(f"Warning: Unknown template type '{low_template_type}', using 'minimal'")
            low_template_type = "minimal"
        
        template = ZERO_SHOT_TEMPLATES[low_template_type]
        print(f"Low-quality template type: {low_template_type}")
        
        # 生成低质量 Prompt
        if template == "":
            low_quality_prompts = ["X" for _ in classnames]  # 使用 "X" 代替空字符串
            print("Using 'X' as low-quality prompt")
        else:
            low_quality_prompts = [template.format(**{"class": cls}) for cls in classnames]
            print(f"Low-quality prompt examples (first 3):")
            for cls, prompt in zip(classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        # 预计算低质量特征
        with torch.no_grad():
            low_tokenized = torch.cat([self.tokenizer(p) for p in low_quality_prompts])
            low_text_features = biomedclip_model_temp.encode_text(low_tokenized.cuda())
        
        self.fixed_low_embeddings = low_text_features  # 低质量特征（冻结）
        print(f"[OK] Low-quality Prompt initialized\n")
        
        # 保存 token 前缀和后缀
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMEDAP.CLASS_TOKEN_POSITION

    def forward(self):
        """构造完整的 Prompt"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    

class CLIP_Inplanted(nn.Module):
    """带Visual Prompt和跨模态交互的图像编码器"""
    def __init__(self, clip_model, enable_fusion=False, fusion_layers=[5, 8]):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.text.transformer.dtype
        self.num_tokens = 4
        self.prompt_dim = 768
        self.prompt_embeddings = torch.zeros(1, self.num_tokens, self.prompt_dim)
        self.deep_prompt_embeddings = torch.zeros(12, self.num_tokens, self.prompt_dim)
        self.prompt_dropout = nn.Dropout(0.5)
        
        # ========== 【修改】动态获取text prompt的维度 ==========
        # BiomedCLIP的text embedding维度
        text_dim = clip_model.text.transformer.embeddings.word_embeddings.embedding_dim
        print(f"[FUSION] Detected visual_dim={self.prompt_dim}, text_dim={text_dim}")

        # ========== 【新增】跨模态融合配置 ==========
        self.enable_fusion = enable_fusion
        self.fusion_layers = fusion_layers  # 在哪些层进行融合
        
        if enable_fusion:
            print(f"[FUSION] Enabling Cross-Modal Fusion at layers: {fusion_layers}")
            # 为每个融合层创建独立的融合模块
            self.fusion_modules = nn.ModuleDict({
                str(layer): CrossModalPromptFusion(
                    visual_dim=self.prompt_dim,  # Visual Prompt: 768维
                    text_dim=text_dim,           # Text Prompt: 动态获取
                    num_heads=8, 
                    dropout=0.1)
                for layer in fusion_layers
            })
            print(f"[FUSION] Created {len(self.fusion_modules)} fusion modules")
        
        # ========== 【新增】存储中间层的visual prompts ==========
        self.visual_prompts_cache = {}

    def forward(self, x, text_prompts=None, label=None):
        """
        注入Visual Prompt并与Text Prompt交互的前向传播
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            text_prompts: Text Prompt特征 [n_cls, n_ctx, 768] (可选)
            label: 真实标签 [B] (训练时使用)
        """
        x = self.image_encoder.trunk.patch_embed(x)
        x = self.image_encoder.trunk._pos_embed(x)
        x = self.image_encoder.trunk.patch_drop(x)
        x = self.image_encoder.trunk.norm_pre(x)
        
        # 注入浅层Visual Prompt
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1).cuda()),
            x[:, 1+self.num_tokens:, :]
        ), dim=1)

        # ========== 【修改】根据训练/测试选择不同策略 ==========
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
                x = self.image_encoder.trunk.blocks[i](x)
            
            # 最终投影
            x = self.image_encoder.trunk.norm(x)
            x = x[:, 0]
            x = self.image_encoder.trunk.fc_norm(x)
            x = self.image_encoder.trunk.head_drop(x)
            x = self.image_encoder.trunk.head(x)
            x = self.image_encoder.head(x)
            return x
        # else:
        #     # ========== 【修改】测试时：使用平均text prompts ==========
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
                
        #         # ========== 【关键改进】测试时使用所有类别的平均text prompts ==========
        #         if self.enable_fusion and i in self.fusion_layers and text_prompts is not None:
        #             curr_visual_prompts = x[:, 1:1+self.num_tokens, :]
        #             # ✅ 使用平均text prompts
        #             curr_text_prompts = text_prompts.mean(dim=0).unsqueeze(0).expand(B, -1, -1)
                    
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
        #         x = self.image_encoder.trunk.blocks[i](x)
            
        #     # 最终投影
        #     x = self.image_encoder.trunk.norm(x)
        #     x = x[:, 0]
        #     x = self.image_encoder.trunk.fc_norm(x)
        #     x = self.image_encoder.trunk.head_drop(x)
        #     x = self.image_encoder.trunk.head(x)
        #     x = self.image_encoder.head(x)
        #     return x

        # ========== 【修改】测试模式 (Inference) ==========
        else:
            # 🔴 【在此处修改策略以运行消融实验】 🔴
            # 可选值: 'mean' (默认), 'null', 'retrieval', 'oracle'
            TEST_STRATEGY = 'oracle'  
            
            # print(f"[DEBUG] Using Inference Strategy: {TEST_STRATEGY}") # 调试时可打开

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
                
                # ========== 融合逻辑修改 ==========
                if self.enable_fusion and i in self.fusion_layers and text_prompts is not None:
                    curr_visual_prompts = x[:, 1:1+self.num_tokens, :]
                    
                    # -------------------------------------------------------
                    # 核心修改：根据策略选择不同的 Text Context
                    # -------------------------------------------------------
                    if TEST_STRATEGY == 'mean':
                        # 策略1: Mean (Ours) - 全局平均，提供领域先验 (Domain Prior)
                        # [N_cls, N_ctx, Dim] -> [1, N_ctx, Dim] -> [B, N_ctx, Dim]
                        curr_text_prompts = text_prompts.mean(dim=0).unsqueeze(0).expand(B, -1, -1)
                        
                    elif TEST_STRATEGY == 'null':
                        # 策略2: Null - 全零向量，阻断文本信息，只靠视觉
                        curr_text_prompts = torch.zeros_like(text_prompts[:1]).expand(B, -1, -1)
                        
                    elif TEST_STRATEGY == 'oracle':
                        # 策略3: Oracle - 使用真实标签 (理论上限)
                        # 注意：如果测试代码没传 label，这里会报错
                        if label is None:
                            # 如果没传 label (常见于 evaluation 脚本)，回退到 mean 并警告
                            # print("Warning: Label is None for Oracle strategy, falling back to Mean")
                            curr_text_prompts = text_prompts.mean(dim=0).unsqueeze(0).expand(B, -1, -1)
                        else:
                            curr_text_prompts = text_prompts[label]

                    elif TEST_STRATEGY == 'retrieval':
                        # 策略4: Retrieval - 基于当前视觉特征的 Top-1 检索
                        # 用于证明：早期视觉特征不成熟，检索容易出错
                        
                        # 1. 归一化当前视觉特征 (CLS token) [B, Dim]
                        vis_feat = x[:, 0]
                        vis_feat = vis_feat / (vis_feat.norm(dim=-1, keepdim=True) + 1e-8)
                        
                        # 2. 归一化所有 Text Contexts (取平均代表该类语义) [N_cls, Dim]
                        txt_feats = text_prompts.mean(dim=1)
                        txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-8)
                        
                        # 3. 计算相似度 [B, N_cls]
                        sim = vis_feat @ txt_feats.t()
                        
                        # 4. 检索 Top-1 索引 [B]
                        top_idx = torch.argmax(sim, dim=1)
                        
                        # 5. 选取对应的 Context [B, N_ctx, Dim]
                        curr_text_prompts = text_prompts[top_idx]
                    
                    else:
                        raise ValueError(f"Unknown TEST_STRATEGY: {TEST_STRATEGY}")
                    # -------------------------------------------------------

                    # 执行融合
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
                    
                    # self.visual_prompts_cache[f'layer_{i}_text'] = enhanced_text
                
                # 执行Transformer block
                x = self.image_encoder.trunk.blocks[i](x)
            
            # 最终投影
            x = self.image_encoder.trunk.norm(x)
            x = x[:, 0]
            x = self.image_encoder.trunk.fc_norm(x)
            x = self.image_encoder.trunk.head_drop(x)
            x = self.image_encoder.trunk.head(x)
            x = self.image_encoder.head(x)
            return x


        # else:
        #     # ========== 【关键改进】测试时：为每个类别分别提取特征 ==========
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
        #             x = self.image_encoder.trunk.blocks[i](x)
                
        #         x = self.image_encoder.trunk.norm(x)
        #         x = x[:, 0]
        #         x = self.image_encoder.trunk.fc_norm(x)
        #         x = self.image_encoder.trunk.head_drop(x)
        #         x = self.image_encoder.trunk.head(x)
        #         x = self.image_encoder.head(x)
        #         return x
            
        #     # ========== 【添加断言】确保进入多类别融合 ==========
        #     assert self.enable_fusion, "[ERROR] Fusion should be enabled in test mode!"
        #     assert text_prompts is not None, "[ERROR] text_prompts is None!"
        #     print(f"[INFO] Test mode: Multi-class fusion activated, n_cls={text_prompts.shape[0]}")
        #     # ========== 如果启用融合，为每个类别提取特征 ==========
        #     n_cls = text_prompts.shape[0]
        #     all_image_features = []
            
        #     for c in range(n_cls):
        #         # 为第c个类别提取特征
        #         x_c = x.clone()  # 克隆输入（避免污染）
                
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
        #             x_c = self.image_encoder.trunk.blocks[i](x_c)
                
        #         # 最终投影
        #         x_c = self.image_encoder.trunk.norm(x_c)
        #         x_c = x_c[:, 0]
        #         x_c = self.image_encoder.trunk.fc_norm(x_c)
        #         x_c = self.image_encoder.trunk.head_drop(x_c)
        #         x_c = self.image_encoder.trunk.head(x_c)
        #         x_c = self.image_encoder.head(x_c)
                
        #         all_image_features.append(x_c.unsqueeze(1))  # [B, 1, dim]
            
        #     # 堆叠所有类别的特征
        #     all_image_features = torch.cat(all_image_features, dim=1)  # [B, n_cls, dim]
            
        #     # 缓存供CustomCLIP使用
        #     print(f"[DEBUG] Writing to cache: shape={all_image_features.shape}")
        #     self.visual_prompts_cache['all_cls_features'] = all_image_features
            
        #     # 返回第一个类别的特征（兼容原接口）
        #     return all_image_features[:, 0, :]


class CustomCLIP(nn.Module):
    """自定义CLIP模型(添加了跨模态融合)"""
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # ========== 【修改】传入融合配置 ==========
        enable_fusion = cfg.TRAINER.BIOMEDAP.ENABLE_FUSION if hasattr(cfg.TRAINER.BIOMEDAP, 'ENABLE_FUSION') else False
        fusion_layers = cfg.TRAINER.BIOMEDAP.FUSION_LAYERS if hasattr(cfg.TRAINER.BIOMEDAP, 'FUSION_LAYERS') else [5, 8]
        
        self.image_encoder = CLIP_Inplanted(
            biomedclip_model,
            enable_fusion=enable_fusion,
            fusion_layers=fusion_layers
        )
        
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # ========== 【新增】Prompt对齐损失权重 ==========
        self.alignment_lambda = cfg.TRAINER.BIOMEDAP.ALIGNMENT_LAMBDA if hasattr(cfg.TRAINER.BIOMEDAP, 'ALIGNMENT_LAMBDA') else 0.0

    def forward(self, image, label=None):
        """前向传播(添加跨模态交互)"""
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # ========== 【修改】获取text prompts的中间表示 ==========
        prompts = self.prompt_learner()  # [n_cls, seq_len, 768]
        
        # 提取text context部分(去掉prefix和suffix)
        # 假设context在中间, 格式为: [SOS] + [CTX] + [CLASS] + [EOS]
        n_ctx = self.prompt_learner.n_ctx
        text_ctx = prompts[:, 1:1+n_ctx, :]  # [n_cls, n_ctx, 768]
        
        # ========== 【修改】提取图像特征(传入text prompts) ==========
        image_features = self.image_encoder(
            image.type(self.dtype),
            text_prompts=text_ctx,  # 传入text prompts
            label=label  # 传入label
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
        
        # 计算logits
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:         
            # 损失1: 交叉熵
            loss_ce = F.cross_entropy(logits, label)
            
            # 损失2: L1对齐(可学习→高质量)
            loss_l1_high = F.l1_loss(
                text_features, 
                fixed_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_HIGH 
            
            # 损失3: KL散度(知识蒸馏)
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel() * self.cfg.TRAINER.BIOMEDAP.KL_LAMBDA

            # 损失4: L1鲁棒性约束(可学习→低质量)
            loss_l1_low = F.l1_loss(
                text_features, 
                fixed_low_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_LOW

            # ========== 【新增】损失5: Prompt对齐损失 ==========
            loss_alignment = 0.0
            if self.alignment_lambda > 0 and hasattr(self.image_encoder, 'visual_prompts_cache'):
                cache = self.image_encoder.visual_prompts_cache
                if len(cache) > 0:
                    # 从缓存中提取enhanced text prompts
                    for key, enhanced_text in cache.items():
                        # enhanced_text: [B, n_ctx, 768]
                        # text_ctx[label]: [B, n_ctx, 768]
                        curr_text = text_ctx[label]
                        
                        # 池化为单个向量
                        enhanced_pooled = enhanced_text.mean(dim=1)  # [B, 768]
                        curr_pooled = curr_text.mean(dim=1)          # [B, 768]
                        
                        # 归一化
                        enhanced_pooled = F.normalize(enhanced_pooled, dim=-1)
                        curr_pooled = F.normalize(curr_pooled, dim=-1)
                        
                        # 计算余弦相似度(最大化)
                        alignment_sim = (enhanced_pooled * curr_pooled).sum(dim=-1).mean()
                        loss_alignment += (1 - alignment_sim)  # 转换为损失
                    
                    loss_alignment /= len(cache)
                    loss_alignment *= self.alignment_lambda
            
            # 总损失
            total_loss = loss_ce + loss_l1_high + loss_kl + loss_l1_low + loss_alignment
            
            return logits, total_loss
        else:
            return logits

        # if self.prompt_learner.training:
        #     # ========== 训练时：标准流程 ==========
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        #     zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        #     logits = logit_scale * image_features @ text_features.t()
            
        #     loss_ce = F.cross_entropy(logits, label)
        #     loss_l1_high = F.l1_loss(text_features, fixed_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_HIGH
        #     loss_kl = F.kl_div(F.log_softmax(logits, dim=1), F.log_softmax(zero_shot_logits, dim=1), reduction='sum', log_target=True) / logits.numel() * self.cfg.TRAINER.BIOMEDAP.KL_LAMBDA
        #     loss_l1_low = F.l1_loss(text_features, fixed_low_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_LOW
            
        #     loss_alignment = 0.0
        #     if self.alignment_lambda > 0 and hasattr(self.image_encoder, 'visual_prompts_cache'):
        #         cache = self.image_encoder.visual_prompts_cache
        #         if len(cache) > 0:
        #             for key, enhanced_text in cache.items():
        #                 if 'layer' in key:
        #                     curr_text = text_ctx[label]
        #                     enhanced_pooled = F.normalize(enhanced_text.mean(dim=1), dim=-1)
        #                     curr_pooled = F.normalize(curr_text.mean(dim=1), dim=-1)
        #                     alignment_sim = (enhanced_pooled * curr_pooled).sum(dim=-1).mean()
        #                     loss_alignment += (1 - alignment_sim)
        #             loss_alignment /= len([k for k in cache.keys() if 'layer' in k])
        #             loss_alignment *= self.alignment_lambda
            
        #     total_loss = loss_ce + loss_l1_high + loss_kl + loss_l1_low + loss_alignment
        #     return logits, total_loss
        
        # else:
        #     # 【添加断言】确保使用多类别特征
        #     assert hasattr(self.image_encoder, 'visual_prompts_cache'), "[ERROR] Cache not found!"

        #     # ========== 【关键改进】测试时：使用多类别特征 ==========
        #     if 'all_cls_features' in self.image_encoder.visual_prompts_cache:
        #         print("[INFO] Using multi-class features from cache")
        #         # 使用每个类别单独提取的特征
        #         all_image_features = self.image_encoder.visual_prompts_cache['all_cls_features']  # [B, n_cls, dim]
        #         all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        #         print(f"[INFO] Using multi-class features, shape: {all_image_features.shape}")
                
        #         # 计算logits（逐类别点积）
        #         logits = logit_scale * (all_image_features * text_features.unsqueeze(0)).sum(dim=-1)  # [B, n_cls]
        #     else:
        #         # 回退方案
        #         print("[WARNING] all_cls_features not in cache, using fallback!")
        #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #         logits = logit_scale * image_features @ text_features.t()
            
        #     return logits



@TRAINER_REGISTRY.register()
class BiomedAP_BiomedCLIP(TrainerX):
    """训练器"""
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        if cfg.TRAINER.BIOMEDAP.PREC == "fp32" or cfg.TRAINER.BIOMEDAP.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP with Cross-Modal Fusion")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        
        # ========== 【修改】添加fusion_modules到可训练参数 ==========
        names_to_update = ["prompt_learner.ctx"]
        
        # 如果启用了融合,添加融合模块参数
        if hasattr(cfg.TRAINER.BIOMEDAP, 'ENABLE_FUSION') and cfg.TRAINER.BIOMEDAP.ENABLE_FUSION:
            for name, param in self.model.named_parameters():
                if "fusion_modules" in name:
                    names_to_update.append(name)
                    param.requires_grad_(True)
        
        for name, param in self.model.named_parameters():
            if not any(update_name in name for update_name in names_to_update):
                param.requires_grad_(False)
        
        # 再次检查
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

            # if not osp.exists(model_path):
            #     raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            if not osp.exists(model_path):
                print(f"No pretrained model found at '{model_path}', training from scratch")
                return

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 忽略固定的 token 向量
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
