"""
BiomedAP (PMC-CLIP backbone)
========================
Adaptive Prompt Learning with Cross-Modal Fusion (PMC-CLIP版本)

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
import os
import os.path as osp
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import requests
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

from trainers.prompt_templates import BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import CUSTOM_BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import ZERO_SHOT_TEMPLATES
from trainers.cross_modal_fusion import CrossModalPromptFusion, LightweightCrossModalFusion

from clip.pmcclip import ModifiedResNet
from transformers import AutoTokenizer, AutoModel

directory = "clip/checkpoints"
files = {
    "text_encoder.pth": "clip/checkpoints/text_encoder.pth",
    "image_encoder(resnet50).pth": "clip/checkpoints/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "clip/checkpoints/text_projection_layer.pth",
}


def download_file(url, filepath):
    """下载文件"""
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts, tokenized_prompts):
        output = self.text_encoder(inputs_embeds=prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer.cuda()
        return text_feature


class PromptLearner(nn.Module):
    """Prompt 学习器（集成高质量 + 低质量特征预计算）"""
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDAP.CTX_INIT
        dtype = torch.float32
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = AutoTokenizer.from_pretrained('clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # ========== 【修改】获取模型所在设备 ==========
        device = next(pmcclip_model.text_encoder.parameters()).device
        print(f"[INIT] text_encoder device: {device}")

        # 初始化可学习上下文向量
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
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
        name_lens = [len(self.tokenizer(name, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']) for name in classnames]
        
        # 使用中等质量模板
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')

        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(tokenized_prompts['input_ids'].cuda()).type(dtype)
            
            # ========== 【修改】确保 text_projection_layer 在正确设备上 ==========
            text_projection_layer = pmcclip_model.text_projection_layer.to(device)
            print(f"[INIT] text_projection_layer device: {text_projection_layer.device}, shape: {text_projection_layer.shape}")

            # ========== 预计算高质量特征（50个模板）==========
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDAP.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids'] for classname in classnames])
                x_tokenized_attn_masks = torch.cat([self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['attention_mask'] for classname in classnames])
                text_features = pmcclip_model.text_encoder(x_tokenized.cuda(), x_tokenized_attn_masks.cuda())
                pooler_output = text_features.pooler_output
                text_features = pooler_output @ pmcclip_model.text_projection_layer.cuda()
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
            low_quality_prompts = [template.format(**{"class": cls}) for cls in classnames]
            print(f"Low-quality prompt examples (first 3):")
            for cls, prompt in zip(classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        # 预计算低质量特征
        with torch.no_grad():
            low_tokenized = torch.cat([self.tokenizer(p if p else "X", padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids'] for p in low_quality_prompts])
            low_tokenized_attn_masks = torch.cat([self.tokenizer(p if p else "X", padding='max_length', truncation=True, max_length=77, return_tensors='pt')['attention_mask'] for p in low_quality_prompts])
            low_text_features = pmcclip_model.text_encoder(low_tokenized.cuda(), low_tokenized_attn_masks.cuda())
            pooler_output = low_text_features.pooler_output
            low_text_features = pooler_output @ pmcclip_model.text_projection_layer.cuda()
        
        self.fixed_low_embeddings = low_text_features  # [N, 512]
        print(f"[OK] Low-quality Prompt initialized\n")

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


class PMCCLIP(nn.Module):
    """PMC-CLIP 模型"""
    def __init__(self, image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained('clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract')

    def forward(self, image, text):
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']
        return image_feature, text_feature


class CLIP_Inplanted(nn.Module):
    """带Visual Prompt和跨模态交互的图像编码器"""
    def __init__(self, clip_model, enable_fusion=False, fusion_layers=[2, 3]):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.image_encoder
        self.dtype = torch.float32
        self.num_tokens = 4
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, 56, 56))
        self.deep_prompt_embeddings_1 = nn.Parameter(torch.zeros(1, self.num_tokens, 56, 56))
        self.deep_prompt_embeddings_2 = nn.Parameter(torch.zeros(1, self.num_tokens, 28, 28))
        self.deep_prompt_embeddings_3 = nn.Parameter(torch.zeros(1, self.num_tokens, 14, 14))
        self.deep_prompt_embeddings_4 = nn.Parameter(torch.zeros(1, self.num_tokens, 7, 7))
        self.prompt_dropout = nn.Dropout(0.5)

        # ========== 【修复】不降维，保持 768 维 ==========
        self.visual_projection = nn.Linear(768, 768, bias=False)
        nn.init.eye_(self.visual_projection.weight)  # 恒等初始化
        print(f"[INIT] Added visual_projection: 768 -> 768 (identity-like)")

        # ========== 【修改】动态获取text prompt的维度 ==========
        # PMC-CLIP的text encoder是BiomedBERT，维度通常是768
        text_dim = clip_model.text_encoder.config.hidden_size
        print(f"[FUSION] Detected text_dim={text_dim}")
        
        # ========== 跨模态融合配置 ==========
        self.enable_fusion = enable_fusion
        self.fusion_layers = fusion_layers  # ResNet层索引: [1,2,3,4]
        
        if enable_fusion:
            print(f"[FUSION] Enabling Cross-Modal Fusion at layers: {fusion_layers}")
            # 为每个融合层创建独立的融合模块
            # 注意: PMC-CLIP使用ResNet50, 不同层的空间维度不同
            fusion_dims = {
                1: 256,   # layer1输出: 256通道, 56x56
                2: 512,   # layer2输出: 512通道, 28x28
                3: 1024,  # layer3输出: 1024通道, 14x14
                4: 2048   # layer4输出: 2048通道, 7x7
            }
            
            self.fusion_modules = nn.ModuleDict({
                str(layer): CrossModalPromptFusion(
                    visual_dim=fusion_dims[layer],  # ✅ Visual特征维度（不同层不同）
                    text_dim=text_dim,              # ✅ Text Prompt维度（768）
                    num_heads=8,
                    dropout=0.1
                )
                for layer in fusion_layers
            })
            print(f"[FUSION] Created {len(self.fusion_modules)} fusion modules")
        
        self.visual_prompts_cache = {}

    def forward(self, x, text_prompts=None, label=None):
        """
        注入Visual Prompt并与Text Prompt交互的前向传播
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            text_prompts: Text Prompt特征 [n_cls, n_ctx, 768] (可选)
            label: 真实标签 [B] (训练时使用)
        """
        # 初始卷积层
        x = self.image_encoder.relu1(self.image_encoder.bn1(self.image_encoder.conv1(x)))
        x = self.image_encoder.relu2(self.image_encoder.bn2(self.image_encoder.conv2(x)))
        x = self.image_encoder.relu3(self.image_encoder.bn3(self.image_encoder.conv3(x)))
        x = self.image_encoder.avgpool(x)
        
        # 注入浅层Visual Prompt
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1, -1)),
            x[:, 1+self.num_tokens:, :]
        ), dim=1)
        
        # ========== 根据训练/测试选择不同策略 ==========
        if self.training and label is not None:
            # ========== 训练时：标准流程 ==========
            # Layer1
            x = self.image_encoder.layer1(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_1.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            # 在layer1后融合
            if self.enable_fusion and 1 in self.fusion_layers and text_prompts is not None:
                x = self._fuse_at_layer(x, text_prompts, label, layer=1)
            
            # Layer2
            x = self.image_encoder.layer2(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_2.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            if self.enable_fusion and 2 in self.fusion_layers and text_prompts is not None:
                x = self._fuse_at_layer(x, text_prompts, label, layer=2)
            
            # Layer3
            x = self.image_encoder.layer3(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_3.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            if self.enable_fusion and 3 in self.fusion_layers and text_prompts is not None:
                x = self._fuse_at_layer(x, text_prompts, label, layer=3)
            
            # Layer4
            x = self.image_encoder.layer4(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_4.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            if self.enable_fusion and 4 in self.fusion_layers and text_prompts is not None:
                x = self._fuse_at_layer(x, text_prompts, label, layer=4)
            
            # 最终池化
            x = self.image_encoder.attnpool(x)

            # ========== 【新增】投影到 512 维 ==========
            x = self.visual_projection(x)  # [B, 768] -> [B, 512]
            return x
        
        else:
            # ========== 【修改】测试时：使用平均text prompts ==========
            # Layer1
            x = self.image_encoder.layer1(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_1.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            # ========== 【关键改进】使用所有类别的平均text prompts ==========
            if self.enable_fusion and 1 in self.fusion_layers and text_prompts is not None:
                # 使用平均text prompts替代单个类别
                avg_text_prompts = text_prompts.mean(dim=0, keepdim=True)  # [1, n_ctx, 768]
                x = self._fuse_at_layer(x, avg_text_prompts, None, layer=1, test_mode=True)
            
            # Layer2
            x = self.image_encoder.layer2(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_2.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            if self.enable_fusion and 2 in self.fusion_layers and text_prompts is not None:
                avg_text_prompts = text_prompts.mean(dim=0, keepdim=True)
                x = self._fuse_at_layer(x, avg_text_prompts, None, layer=2, test_mode=True)
            
            # Layer3
            x = self.image_encoder.layer3(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_3.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            if self.enable_fusion and 3 in self.fusion_layers and text_prompts is not None:
                avg_text_prompts = text_prompts.mean(dim=0, keepdim=True)
                x = self._fuse_at_layer(x, avg_text_prompts, None, layer=3, test_mode=True)
            
            # Layer4
            x = self.image_encoder.layer4(x)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings_4.expand(B, -1, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            
            if self.enable_fusion and 4 in self.fusion_layers and text_prompts is not None:
                avg_text_prompts = text_prompts.mean(dim=0, keepdim=True)
                x = self._fuse_at_layer(x, avg_text_prompts, None, layer=4, test_mode=True)
            
            # 最终池化
            x = self.image_encoder.attnpool(x)
            x = self.visual_projection(x)
            return x

            # # ========== 测试时：不同策略 ==========
            # if not self.enable_fusion or text_prompts is None:
            #     # 如果未启用融合，直接走标准流程
            #     x = self.image_encoder.layer1(x)
            #     x = torch.cat((x[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_1.expand(B, -1, -1, -1)), x[:, 1+self.num_tokens:, :]), dim=1)
            #     x = self.image_encoder.layer2(x)
            #     x = torch.cat((x[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_2.expand(B, -1, -1, -1)), x[:, 1+self.num_tokens:, :]), dim=1)
            #     x = self.image_encoder.layer3(x)
            #     x = torch.cat((x[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_3.expand(B, -1, -1, -1)), x[:, 1+self.num_tokens:, :]), dim=1)
            #     x = self.image_encoder.layer4(x)
            #     x = torch.cat((x[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_4.expand(B, -1, -1, -1)), x[:, 1+self.num_tokens:, :]), dim=1)
            #     x = self.image_encoder.attnpool(x)
            #     return x
            
            # # ========== 测试时：为每个类别分别提取特征 ==========
            # assert self.enable_fusion, "[ERROR] Fusion should be enabled in test mode!"
            # assert text_prompts is not None, "[ERROR] text_prompts is None!"
            # print(f"[INFO] Test mode: Multi-class fusion activated, n_cls={text_prompts.shape[0]}")
            
            # n_cls = text_prompts.shape[0]
            # all_image_features = []
            
            # for c in range(n_cls):
            #     # 为第c个类别提取特征
            #     x_c = x.clone()
                
            #     # Layer1
            #     x_c = self.image_encoder.layer1(x_c)
            #     x_c = torch.cat((x_c[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_1.expand(B, -1, -1, -1)), x_c[:, 1+self.num_tokens:, :]), dim=1)
            #     if 1 in self.fusion_layers:
            #         x_c = self._fuse_at_layer(x_c, text_prompts[c:c+1], None, layer=1, test_mode=True)
                
            #     # Layer2
            #     x_c = self.image_encoder.layer2(x_c)
            #     x_c = torch.cat((x_c[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_2.expand(B, -1, -1, -1)), x_c[:, 1+self.num_tokens:, :]), dim=1)
            #     if 2 in self.fusion_layers:
            #         x_c = self._fuse_at_layer(x_c, text_prompts[c:c+1], None, layer=2, test_mode=True)
                
            #     # Layer3
            #     x_c = self.image_encoder.layer3(x_c)
            #     x_c = torch.cat((x_c[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_3.expand(B, -1, -1, -1)), x_c[:, 1+self.num_tokens:, :]), dim=1)
            #     if 3 in self.fusion_layers:
            #         x_c = self._fuse_at_layer(x_c, text_prompts[c:c+1], None, layer=3, test_mode=True)
                
            #     # Layer4
            #     x_c = self.image_encoder.layer4(x_c)
            #     x_c = torch.cat((x_c[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings_4.expand(B, -1, -1, -1)), x_c[:, 1+self.num_tokens:, :]), dim=1)
            #     if 4 in self.fusion_layers:
            #         x_c = self._fuse_at_layer(x_c, text_prompts[c:c+1], None, layer=4, test_mode=True)
                
            #     # 最终池化
            #     x_c = self.image_encoder.attnpool(x_c)
            #     all_image_features.append(x_c.unsqueeze(1))  # [B, 1, dim]
            
            # # 堆叠所有类别的特征
            # all_image_features = torch.cat(all_image_features, dim=1)  # [B, n_cls, dim]
            
            # # 缓存供CustomCLIP使用
            # print(f"[DEBUG] Writing to cache: shape={all_image_features.shape}")
            # self.visual_prompts_cache['all_cls_features'] = all_image_features
            
            # # 返回第一个类别的特征（兼容原接口）
            # return all_image_features[:, 0, :]
    
    def _fuse_at_layer(self, x, text_prompts, label, layer, test_mode=False):
        """
        在指定层执行跨模态融合
        
        Args:
            x: 当前特征图 [B, C, H, W]
            text_prompts: Text Prompt特征 [n_cls, n_ctx, 768] 或 [1, n_ctx, 768] (test_mode)
            label: 真实标签 [B] (训练时使用)
            layer: 当前层索引
            test_mode: 是否为测试模式
        """
        B, C, H, W = x.shape
        
        # 提取当前的visual prompts (假设是第1到num_tokens的通道)
        curr_visual_prompts = x[:, 1:1+self.num_tokens, :, :]  # [B, num_tokens, H, W]
        
        # 展平空间维度: [B, num_tokens, H, W] -> [B, num_tokens, H*W] -> [B, num_tokens*H*W]
        curr_visual_prompts_flat = curr_visual_prompts.flatten(2).flatten(1)  # [B, num_tokens*H*W]
        
        # ========== 【修改】获取当前层的visual维度 ==========
        fusion_dims = {1: 256, 2: 512, 3: 1024, 4: 2048}
        current_visual_dim = fusion_dims[layer]
        
        # 投影到当前层的visual维度
        if not hasattr(self, f'_proj_layer{layer}'):
            setattr(self, f'_proj_layer{layer}', 
                    nn.Linear(self.num_tokens * H * W, current_visual_dim).cuda())
        proj = getattr(self, f'_proj_layer{layer}')
        
        curr_visual_prompts_proj = proj(curr_visual_prompts_flat)  # [B, current_visual_dim]
        curr_visual_prompts_proj = curr_visual_prompts_proj.unsqueeze(1).expand(-1, self.num_tokens, -1)  # [B, num_tokens, current_visual_dim]
        
        # 获取对应的text prompts
        if test_mode:
            curr_text_prompts = text_prompts.expand(B, -1, -1)  # [B, n_ctx, 768]
        else:
            curr_text_prompts = text_prompts[label]  # [B, n_ctx, 768]
        
        # ========== 执行跨模态融合（内部自动处理维度对齐）==========
        fusion_module = self.fusion_modules[str(layer)]
        enhanced_visual, enhanced_text = fusion_module(
            curr_visual_prompts_proj,  # [B, num_tokens, current_visual_dim]
            curr_text_prompts          # [B, n_ctx, 768]
        )
        
        # 将增强后的visual prompts投影回空间维度
        if not hasattr(self, f'_inv_proj_layer{layer}'):
            setattr(self, f'_inv_proj_layer{layer}',
                    nn.Linear(current_visual_dim, self.num_tokens * H * W).cuda())
        inv_proj = getattr(self, f'_inv_proj_layer{layer}')
        
        enhanced_visual_flat = inv_proj(enhanced_visual.mean(dim=1))  # [B, num_tokens*H*W]
        enhanced_visual_spatial = enhanced_visual_flat.view(B, self.num_tokens, H, W)  # [B, num_tokens, H, W]
        
        # 更新特征图
        x = torch.cat([
            x[:, :1, :, :],                    # 保留第一个通道
            enhanced_visual_spatial,           # 增强后的visual prompts
            x[:, 1+self.num_tokens:, :, :]     # 剩余通道
        ], dim=1)
        
        # 缓存enhanced text prompts
        if not test_mode:
            self.visual_prompts_cache[f'layer_{layer}_text'] = enhanced_text
        
        return x


class CustomCLIP(nn.Module):
    """自定义 CLIP 模型（集成跨模态融合 + 5个损失函数）"""
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, pmcclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.cfg = cfg
        
        # ========== 传入融合配置 ==========
        enable_fusion = cfg.TRAINER.BIOMEDAP.ENABLE_FUSION if hasattr(cfg.TRAINER.BIOMEDAP, 'ENABLE_FUSION') else False
        fusion_layers = cfg.TRAINER.BIOMEDAP.FUSION_LAYERS if hasattr(cfg.TRAINER.BIOMEDAP, 'FUSION_LAYERS') else [2, 3]
        
        self.image_encoder = CLIP_Inplanted(
            pmcclip_model,
            enable_fusion=enable_fusion,
            fusion_layers=fusion_layers
        )
        
        self.text_encoder = TextEncoder(pmcclip_model)
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # ========== Prompt对齐损失权重 ==========
        self.alignment_lambda = cfg.TRAINER.BIOMEDAP.ALIGNMENT_LAMBDA if hasattr(cfg.TRAINER.BIOMEDAP, 'ALIGNMENT_LAMBDA') else 0.0

    def forward(self, image, label=None):
        """前向传播(添加跨模态交互)"""
        tokenized_prompts = self.tokenized_prompts
        logit_scale = math.exp(self.logit_scale)

        # ========== 获取text prompts的中间表示 ==========
        prompts = self.prompt_learner()  # [n_cls, seq_len, 768]
        
        # 提取text context部分(去掉prefix和suffix)
        n_ctx = self.prompt_learner.n_ctx
        text_ctx = prompts[:, 1:1+n_ctx, :]  # [n_cls, n_ctx, 768]
        
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
            # 在 CustomCLIP.forward 的训练分支中 (约第 621-628 行)
            print(f"[DEBUG] image_features.shape = {image_features.shape}")
            print(f"[DEBUG] text_features.shape = {text_features.shape}")
            print(f"[DEBUG] fixed_embeddings.shape = {fixed_embeddings.shape}")

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
class BiomedAP_PMCCLIP(TrainerX):
    """训练器"""
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading PMC-CLIP")
        
        # 下载并加载模型权重
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for filename, filepath in files.items():
            if not os.path.exists(filepath):
                print(f"{filename} not found. Attempting to download...")
                url = f"https://huggingface.co/spaces/microsoft/BiomedCLIP/resolve/main/{filename}"
                download_file(url, filepath)
        
        # 加载图像编码器(ResNet50)
        layers = [3, 4, 6, 3]
        output_dim = 768
        heads = output_dim * 32 // 64

        # ========== 【修改】heads 基于 ResNet50 layer4 的输出维度（2048）==========
        width = 64  # ResNet50 的 width
        input_dim = width * 32  # layer4 输出: 64 * 32 = 2048
        heads = input_dim // 64  # heads = 2048 // 64 = 32 

        image_encoder = ModifiedResNet(layers, output_dim, heads)
        checkpoint = torch.load('clip/checkpoints/image_encoder(resnet50).pth')
        image_encoder.load_state_dict(checkpoint, strict=True)
        
        # 加载文本编码器
        text_encoder = AutoModel.from_pretrained('clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract')
        checkpoint = torch.load('clip/checkpoints/text_encoder.pth')
        text_encoder.load_state_dict(checkpoint, strict=True)
        
        # 加载文本投影层
        projection_layer = torch.load('clip/checkpoints/text_projection_layer.pth')

        # ========== 【新增】验证并修正维度 ==========
        print(f"[INIT] Original projection_layer.shape = {projection_layer.shape}")

        # ========== 【修复】不降维，保持 768 维 ==========
        if projection_layer.shape != (768, 768):
            raise ValueError(f"Expected [768, 768], got {projection_layer.shape}")

        print(f"[INIT] Using [768, 768] projection (no dimensionality reduction)")

        print(f"[INIT] Final projection_layer.shape = {projection_layer.shape}")

        # 创建PMC-CLIP模型
        pmcclip_model = PMCCLIP(image_encoder, text_encoder, projection_layer)

        print(f"[DEVICE] Moving pmcclip_model to {self.device}")
        pmcclip_model = pmcclip_model.to(self.device)

        # ========== 【新增】确保 text_projection_layer 在正确设备上 ==========
        pmcclip_model.text_projection_layer = pmcclip_model.text_projection_layer.to(self.device)
        print(f"[DEVICE] text_projection_layer device: {pmcclip_model.text_projection_layer.device}")
        print(f"[DEVICE] text_projection_layer shape: {pmcclip_model.text_projection_layer.shape}")
        
        if cfg.TRAINER.BIOMEDAP.PREC == "fp32" or cfg.TRAINER.BIOMEDAP.PREC == "amp":
            pmcclip_model.float()

        print("Building custom PMC-CLIP with Cross-Modal Fusion")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model.eval())

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
                logits, loss = model(image, label)
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
