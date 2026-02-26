"""
BiomedTriplePrompt Trainer
==========================
三层级梯度式 Prompt 学习：高质量（教师）→ 中等质量（学生1）→ 低质量（学生2）

核心机制:
1. 高质量 Prompt（教师，冻结）：GPT-4 生成的完整临床描述
2. 中等质量 Prompt（学生1，可学习）：CUSTOM_BIOMEDDPT_TEMPLATES 固定模板
3. 低质量 Prompt（学生2，可学习）：类别名或空提示

知识蒸馏路径:
- 高质量 → 中等质量：细粒度语义传递
- 高质量 → 低质量：强到弱的鲁棒性学习
- 中等质量 → 低质量：中间层语义桥接

文件位置：trainers/BiomedTriplePrompt/biomedtripleprompt.py
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from open_clip import create_model_from_pretrained
from open_clip.tokenizer import tokenize

# 导入 Prompt 模板
from trainers.prompt_templates import (
    TEMPLATES,  # 高质量 GPT-4 Prompt（已有）
    ZERO_SHOT_TEMPLATES  # 低质量 Prompt（新增）
)


# ========== 中等质量 Prompt 模板（CUSTOM_BIOMEDDPT_TEMPLATES）==========
CUSTOM_BIOMEDDPT_TEMPLATES = {
    "BTMRI": "a MR photo of a {} in the brain.",
    "BUSI": "a ultrasound photo of a {} in the breast.",
    "CHMNIST": "a histopathological photo of a {}.",
    "COVID_19": "a chest X-ray photo of a {} affected by COVID-19 in the lung.",
    "CTKidney": "a CT photo of a {} in the kidney.",
    "DermaMNIST": "a dermatoscopy photo of a {} in the skin.",
    "KneeXray": "a frontal X-ray photo of a {} in the knee joint.",
    "Kvasir": "a endoscopic photo of a {} in the colon.",
    "LungColon": "a histopathological photo of a {}.",
    "OCTMNIST": "a OCT photo of a {}.",
    "RETINA": "a photo of a {} presented in image.",
}


def load_biomedclip_to_cpu(cfg):
    """加载 BiomedCLIP 模型"""
    print("Loading BiomedCLIP-PubMedBERT_256-vit_base_patch16_224...")
    clip_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
        clip_model.float()
    
    return clip_model, preprocess


class TextEncoder(nn.Module):
    """文本编码器（BiomedCLIP 的 PubMedBERT）"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text.transformer
        self.token_embedding = clip_model.text.token_embedding
        self.positional_embedding = clip_model.text.positional_embedding
        self.ln_final = clip_model.text.ln_final
        self.text_projection = clip_model.text.text_projection
        self.attn_mask = clip_model.text.attn_mask
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """前向传播"""
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 提取 [EOS] token 特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class TriplePromptLearner(nn.Module):
    """
    三层级 Prompt 学习器
    
    包含:
    1. 高质量 Prompt（教师，冻结）：GPT-4 生成的完整临床描述
    2. 中等质量 Prompt（学生1，可学习）：CUSTOM_BIOMEDDPT_TEMPLATES 固定模板
    3. 低质量 Prompt（学生2，可学习）：类别名或空提示
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.COOP.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.text.ln_final.weight.shape[0]
        self.dataset_name = cfg.DATASET.NAME
        
        print("\n" + "="*80)
        print("🚀 初始化三层级 Prompt 学习器")
        print("="*80)
        
        # ========== 1. 高质量 Prompt（教师，冻结）==========
        print("\n【层级 1】高质量 Prompt（教师，冻结）")
        print("-" * 80)
        self._init_high_quality_prompts(clip_model)
        
        # ========== 2. 中等质量 Prompt（学生1，可学习）==========
        print("\n【层级 2】中等质量 Prompt（学生1，可学习）")
        print("-" * 80)
        self._init_medium_quality_prompts(clip_model)
        
        # ========== 3. 低质量 Prompt（学生2，可学习）==========
        print("\n【层级 3】低质量 Prompt（学生2，可学习）")
        print("-" * 80)
        self._init_low_quality_prompts(clip_model)
        
        print("\n" + "="*80)
        print("✅ 三层级 Prompt 初始化完成")
        print("="*80 + "\n")

    def _init_high_quality_prompts(self, clip_model):
        """初始化高质量 Prompt（GPT-4 生成）"""
        high_quality_prompts = []
        
        for cls in self.classnames:
            if self.dataset_name in TEMPLATES and cls in TEMPLATES[self.dataset_name]:
                # 使用第一条 GPT-4 描述
                prompt = TEMPLATES[self.dataset_name][cls][0]
                high_quality_prompts.append(prompt)
            else:
                print(f"⚠️  警告: 未找到 {self.dataset_name}/{cls} 的 GPT-4 Prompt，使用默认")
                high_quality_prompts.append(f"a medical image of {cls}")
        
        # 分词和嵌入
        self.high_quality_tokenized = torch.cat([
            tokenize([p], context_length=77) for p in high_quality_prompts
        ])
        
        with torch.no_grad():
            high_quality_embedding = clip_model.text.token_embedding(
                self.high_quality_tokenized
            ).type(self.dtype)
        
        # 冻结高质量 Prompt（不参与训练）
        self.register_buffer("high_quality_prompts", high_quality_embedding)
        
        print(f"✅ 加载 {self.n_cls} 个高质量 Prompt（GPT-4 生成）")
        print(f"示例: {high_quality_prompts[0][:65]}...")

    def _init_medium_quality_prompts(self, clip_model):
        """初始化中等质量 Prompt（CUSTOM_BIOMEDDPT_TEMPLATES）"""
        # 获取数据集对应的模板
        if self.dataset_name in CUSTOM_BIOMEDDPT_TEMPLATES:
            template = CUSTOM_BIOMEDDPT_TEMPLATES[self.dataset_name]
        else:
            print(f"⚠️  警告: 未找到 {self.dataset_name} 的中等质量模板，使用默认")
            template = "a medical image of a {}."
        
        # 生成中等质量 Prompt
        medium_quality_prompts = [template.format(cls) for cls in self.classnames]
        
        print(f"使用模板: {template}")
        print(f"生成的中等质量 Prompt 示例:")
        for cls, prompt in zip(self.classnames[:3], medium_quality_prompts[:3]):
            print(f"  {cls:15} -> {prompt}")
        
        # 使用模板初始化可学习向量
        init_text = template.replace("{}", self.classnames[0])
        prompt = tokenize([init_text], context_length=77)[0]
        with torch.no_grad():
            embedding = clip_model.text.token_embedding(prompt).type(self.dtype)
        
        # 提取模板部分（去掉类别名）
        # 例如 "a MR photo of a" 部分
        init_words = init_text.split(self.classnames[0])[0].strip().split()
        n_ctx_actual = min(self.n_ctx, len(init_words))
        ctx_vectors_med = embedding[1: 1 + n_ctx_actual, :]
        
        # 填充到指定长度
        if n_ctx_actual < self.n_ctx:
            padding = torch.zeros(
                self.n_ctx - n_ctx_actual, self.ctx_dim, dtype=self.dtype
            )
            ctx_vectors_med = torch.cat([ctx_vectors_med, padding], dim=0)
        
        # 可学习的中等质量上下文向量
        self.ctx_medium = nn.Parameter(ctx_vectors_med)
        
        # 构造中等质量 Prompt 的固定部分
        self.medium_quality_tokenized = torch.cat([
            tokenize([p], context_length=77) for p in medium_quality_prompts
        ])
        
        with torch.no_grad():
            medium_quality_embedding = clip_model.text.token_embedding(
                self.medium_quality_tokenized
            ).type(self.dtype)
        
        self.register_buffer("token_prefix_med", medium_quality_embedding[:, :1, :])
        self.register_buffer("token_suffix_med", medium_quality_embedding[:, 1 + self.n_ctx:, :])
        
        print(f"✅ 中等质量 Prompt 初始化完成，可学习参数: {self.ctx_medium.numel()}")

    def _init_low_quality_prompts(self, clip_model):
        """初始化低质量 Prompt（类别名或空提示）"""
        # 获取低质量模板类型
        low_template_type = self.cfg.TRAINER.BIOMEDTRIPLEPROMPT.LOW_TEMPLATE_TYPE
        
        if low_template_type not in ZERO_SHOT_TEMPLATES:
            print(f"⚠️  警告: 未知模板类型 '{low_template_type}'，使用 'minimal'")
            low_template_type = "minimal"
        
        template = ZERO_SHOT_TEMPLATES[low_template_type]
        print(f"使用低质量模板: {low_template_type}")
        
        # 生成低质量 Prompt
        if template == "":
            low_quality_prompts = ["" for _ in self.classnames]
            print("使用空字符串作为低质量 Prompt")
        else:
            low_quality_prompts = [template.format(**{"class": cls}) for cls in self.classnames]
            print(f"生成的低质量 Prompt 示例:")
            for cls, prompt in zip(self.classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        # 初始化可学习向量
        if low_quality_prompts[0] == "":
            # 空字符串：随机初始化
            print("使用随机初始化（空提示）")
            ctx_vectors_low = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors_low, std=0.02)
        else:
            # 使用第一个低质量 Prompt 编码
            init_text = low_quality_prompts[0]
            prompt = tokenize([init_text], context_length=77)[0]
            with torch.no_grad():
                embedding = clip_model.text.token_embedding(prompt).type(self.dtype)
            
            init_words = init_text.split()
            n_ctx_actual = min(self.n_ctx, len(init_words))
            ctx_vectors_low = embedding[1: 1 + n_ctx_actual, :]
            
            # 填充
            if n_ctx_actual < self.n_ctx:
                padding = torch.zeros(
                    self.n_ctx - n_ctx_actual, self.ctx_dim, dtype=self.dtype
                )
                ctx_vectors_low = torch.cat([ctx_vectors_low, padding], dim=0)
        
        # 可学习的低质量上下文向量
        self.ctx_low = nn.Parameter(ctx_vectors_low)
        
        # 构造低质量 Prompt 的固定部分
        low_quality_full = [f"{p} ." if p else "X ." for p in low_quality_prompts]
        self.low_quality_tokenized = torch.cat([
            tokenize([p], context_length=77) for p in low_quality_full
        ])
        
        with torch.no_grad():
            low_quality_embedding = clip_model.text.token_embedding(
                self.low_quality_tokenized
            ).type(self.dtype)
        
        self.register_buffer("token_prefix_low", low_quality_embedding[:, :1, :])
        self.register_buffer("token_suffix_low", low_quality_embedding[:, 1 + self.n_ctx:, :])
        
        print(f"[OK] Low-quality Prompt initialized，可学习参数: {self.ctx_low.numel()}")

    def forward(self):
        """
        返回三层级 Prompt 嵌入
        
        返回:
            high_quality_prompts: 高质量 Prompt（冻结）
            medium_quality_prompts: 中等质量 Prompt（可学习）
            low_quality_prompts: 低质量 Prompt（可学习）
        """
        # 1. 高质量 Prompt（冻结，直接返回）
        high_quality_prompts = self.high_quality_prompts
        
        # 2. 中等质量 Prompt（可学习）
        ctx_med = self.ctx_medium
        if ctx_med.dim() == 2:
            ctx_med = ctx_med.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix_med = self.token_prefix_med
        suffix_med = self.token_suffix_med
        medium_quality_prompts = torch.cat([prefix_med, ctx_med, suffix_med], dim=1)
        
        # 3. 低质量 Prompt（可学习）
        ctx_low = self.ctx_low
        if ctx_low.dim() == 2:
            ctx_low = ctx_low.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix_low = self.token_prefix_low
        suffix_low = self.token_suffix_low
        low_quality_prompts = torch.cat([prefix_low, ctx_low, suffix_low], dim=1)
        
        return high_quality_prompts, medium_quality_prompts, low_quality_prompts


class CustomCLIP(nn.Module):
    """三路径 CLIP 模型"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = TriplePromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.high_quality_tokenized = self.prompt_learner.high_quality_tokenized
        self.medium_quality_tokenized = self.prompt_learner.medium_quality_tokenized
        self.low_quality_tokenized = self.prompt_learner.low_quality_tokenized

    def forward(self, image):
        """
        前向传播
        
        返回:
            logits_high: 高质量 Prompt 的 logits
            logits_medium: 中等质量 Prompt 的 logits
            logits_low: 低质量 Prompt 的 logits
        """
        # 提取图像特征
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 获取三层级 Prompt
        high_prompts, medium_prompts, low_prompts = self.prompt_learner()
        
        # 编码高质量 Prompt
        text_features_high = self.text_encoder(high_prompts, self.high_quality_tokenized)
        text_features_high = text_features_high / text_features_high.norm(dim=-1, keepdim=True)
        
        # 编码中等质量 Prompt
        text_features_medium = self.text_encoder(medium_prompts, self.medium_quality_tokenized)
        text_features_medium = text_features_medium / text_features_medium.norm(dim=-1, keepdim=True)
        
        # 编码低质量 Prompt
        text_features_low = self.text_encoder(low_prompts, self.low_quality_tokenized)
        text_features_low = text_features_low / text_features_low.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_high = logit_scale * image_features @ text_features_high.t()
        logits_medium = logit_scale * image_features @ text_features_medium.t()
        logits_low = logit_scale * image_features @ text_features_low.t()
        
        return logits_high, logits_medium, logits_low


@TRAINER_REGISTRY.register()
class BiomedTriplePrompt(TrainerX):
    """
    BiomedTriplePrompt 训练器
    
    三层级梯度式知识蒸馏:
    1. 高质量（教师）→ 中等质量（学生1）：细粒度语义传递
    2. 高质量（教师）→ 低质量（学生2）：强到弱的鲁棒性学习
    3. 中等质量（学生1）→ 低质量（学生2）：中间层语义桥接
    
    损失函数:
    L = L_ce_med + L_ce_low + 
        λ1 * L_kd(high→med) + λ2 * L_kd(high→low) + λ3 * L_kd(med→low) +
        λ4 * L_align(high, med) + λ5 * L_align(high, low)
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """构建三路径模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print(f"\n{'='*80}")
        print(f"🚀 构建 BiomedTriplePrompt 模型")
        print(f"{'='*80}")
        
        # 加载 BiomedCLIP
        clip_model, _ = load_biomedclip_to_cpu(cfg)
        
        # 构建三路径模型
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model.to(self.device)
        
        # 只优化中等质量和低质量 Prompt 的参数
        print("\n🎯 可训练参数:")
        for name, param in self.model.named_parameters():
            if "ctx_medium" in name or "ctx_low" in name:
                print(f"  ✅ {name}: {param.shape}")
            else:
                param.requires_grad = False
        
        # 优化器
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
        
        print(f"\n{'='*80}\n")

    def forward_backward(self, batch):
        """
        训练步骤
        
        损失函数:
        L = L_ce_med + L_ce_low + 
            λ1 * L_kd(high→med) + λ2 * L_kd(high→low) + λ3 * L_kd(med→low) +
            λ4 * L_align(high, med) + λ5 * L_align(high, low)
        """
        image, label = self.parse_batch_train(batch)
        
        # 前向传播
        logits_high, logits_medium, logits_low = self.model(image)
        
        # ========== 损失 1 & 2：中等和低质量路径的交叉熵损失 ==========
        loss_ce_medium = F.cross_entropy(logits_medium, label)
        loss_ce_low = F.cross_entropy(logits_low, label)
        
        # ========== 损失 3：知识蒸馏（高质量 → 中等质量）==========
        T = self.cfg.TRAINER.BIOMEDTRIPLEPROMPT.T
        loss_kd_high_to_med = F.kl_div(
            F.log_softmax(logits_medium / T, dim=1),
            F.softmax(logits_high.detach() / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)
        
        # ========== 损失 4：知识蒸馏（高质量 → 低质量）==========
        loss_kd_high_to_low = F.kl_div(
            F.log_softmax(logits_low / T, dim=1),
            F.softmax(logits_high.detach() / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)
        
        # ========== 损失 5：知识蒸馏（中等质量 → 低质量）==========
        loss_kd_med_to_low = F.kl_div(
            F.log_softmax(logits_low / T, dim=1),
            F.softmax(logits_medium.detach() / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)
        
        # ========== 损失 6 & 7：特征对齐损失 ==========
        high_prompts, medium_prompts, low_prompts = self.model.prompt_learner()
        
        loss_align_high_med = F.mse_loss(medium_prompts, high_prompts.detach())
        loss_align_high_low = F.mse_loss(low_prompts, high_prompts.detach())
        
        # ========== 总损失 ==========
        cfg_tp = self.cfg.TRAINER.BIOMEDTRIPLEPROMPT
        
        loss = (
            loss_ce_medium + loss_ce_low +
            cfg_tp.LAMBDA_KD_HIGH_MED * loss_kd_high_to_med +
            cfg_tp.LAMBDA_KD_HIGH_LOW * loss_kd_high_to_low +
            cfg_tp.LAMBDA_KD_MED_LOW * loss_kd_med_to_low +
            cfg_tp.LAMBDA_ALIGN_HIGH_MED * loss_align_high_med +
            cfg_tp.LAMBDA_ALIGN_HIGH_LOW * loss_align_high_low
        )
        
        # 反向传播
        self.model_backward_and_update(loss)
        
        # 记录损失
        loss_summary = {
            "loss": loss.item(),
            "loss_ce_med": loss_ce_medium.item(),
            "loss_ce_low": loss_ce_low.item(),
            "loss_kd_h2m": loss_kd_high_to_med.item(),
            "loss_kd_h2l": loss_kd_high_to_low.item(),
            "loss_kd_m2l": loss_kd_med_to_low.item(),
            "loss_align_hm": loss_align_high_med.item(),
            "loss_align_hl": loss_align_high_low.item(),
            "acc_high": compute_accuracy(logits_high, label)[0].item(),
            "acc_med": compute_accuracy(logits_medium, label)[0].item(),
            "acc_low": compute_accuracy(logits_low, label)[0].item(),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    def parse_batch_train(self, batch):
        """解析训练批次"""
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None, trainer=None):
        """
        测试（使用低质量 Prompt）
        
        最终目标：让低质量 Prompt 达到接近高质量 Prompt 的性能
        """
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if split is None:
            split = self.cfg.TEST.SPLIT
        
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        
        print(f"🧪 测试低质量 Prompt 性能（{split} split）")
        
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            _, _, logits_low = self.model(input)  # 只使用低质量 Prompt
            self.evaluator.process(logits_low, label)
        
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        return list(results.values())[0]
