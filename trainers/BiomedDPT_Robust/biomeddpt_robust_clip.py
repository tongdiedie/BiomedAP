"""
BiomedDPT_Robust (CLIP backbone)
================================
BiomedDPT + 低质量 Prompt 鲁棒性增强

核心改进：在原有 BiomedDPT_CLIP 基础上添加低质量 Prompt 约束
损失函数：L = L_ce + λ1 * L_L1_high + λ2 * L_KL + λ3 * L_L1_low
"""

# 【关键修复】禁用可能导致 sm80 错误的优化
import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import copy
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
    """Prompt 学习器（在原版基础上添加低质量 Prompt 约束）"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDDPT_ROBUST.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDDPT_ROBUST.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化可学习上下文向量 
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
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
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        clip_model_temp = load_clip_to_cpu(cfg).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg).float().cuda()
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            
            # 预计算高质量特征 
            all_teacher_features = []
            for i in range(cfg.TRAINER.BIOMEDDPT_ROBUST.N_PROMPTS):
                x_tokenized = torch.cat([clip.tokenize(BIOMEDDPT_TEMPLATES[classname][i]) for classname in classnames])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        
        # ========== 【关键新增】预计算低质量特征 ==========
        print("[ANCHOR] Loading low-quality Prompt (robustness anchor)")
        low_template_type = cfg.TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE
        if low_template_type not in ZERO_SHOT_TEMPLATES:
            print(f"Warning: Unknown template type '{low_template_type}', using 'minimal'")
            low_template_type = "minimal"
        template = ZERO_SHOT_TEMPLATES[low_template_type]
        print(f"Low-quality template type: {low_template_type}")
        
        if template == "":
            low_quality_prompts = ["X" for _ in classnames]
            print("Using 'X' as low-quality prompt")
        else:
            low_quality_prompts = [template.format(**{"class": cls}) for cls in classnames]
            print(f"Low-quality prompt examples (first 3):")
            for cls, prompt in zip(classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        with torch.no_grad():
            low_tokenized = torch.cat([clip.tokenize(p if p else "X") for p in low_quality_prompts])
            low_text_features = clip_model_temp.encode_text(low_tokenized.cuda())
        
        self.fixed_low_embeddings = low_text_features
        print(f"[OK] Low-quality Prompt initialized\n")

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CLIP_Inplanted(nn.Module):
    """带 Visual Prompt 的图像编码器 """
    def __init__(self, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.num_tokens = 4
        self.prompt_dim = 768
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.prompt_dim))
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(12, self.num_tokens, self.prompt_dim))
        self.prompt_dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        
        B = x.shape[0]
        x = torch.cat((x[:, :1, :], self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)), x[:, 1+self.num_tokens:, :]), dim=1)
        
        for i in range(12):
            B = x.shape[0]
            x = torch.cat((x[:, :1, :], self.prompt_dropout(self.deep_prompt_embeddings[i].expand(B, -1, -1)), x[:, 1+self.num_tokens:, :]), dim=1)
            x = x.permute(1, 0, 2)
            x = self.image_encoder.transformer.resblocks[i](x)
            x = x.permute(1, 0, 2)
        
        x = self.image_encoder.ln_post(x[:, 0, :])
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj
        return x


class CustomCLIP(nn.Module):
    """自定义 CLIP（在原版基础上添加低质量 Prompt 损失）"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = CLIP_Inplanted(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.cfg = cfg

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 【关键新增】低质量特征
        fixed_low_embeddings = self.prompt_learner.fixed_low_embeddings
        fixed_low_embeddings = fixed_low_embeddings / fixed_low_embeddings.norm(dim=-1, keepdim=True)
        
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:
            loss_ce = F.cross_entropy(logits, label)
            loss_l1 = F.l1_loss(text_features, fixed_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH
            loss_kl = F.kl_div(F.log_softmax(logits, dim=1), F.log_softmax(zero_shot_logits, dim=1), reduction='sum', log_target=True) / logits.numel() * self.cfg.TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA
            
            # 【关键新增】低质量 Prompt 约束
            loss_l1_low = F.l1_loss(text_features, fixed_low_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW
            
            total_loss = loss_ce + loss_l1 + loss_kl + loss_l1_low
            return logits, total_loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class BiomedDPT_Robust_CLIP(TrainerX):
    """训练器（与 biomeddpt_clip.py 完全相同）"""
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDDPT_ROBUST.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BIOMEDDPT_ROBUST.PREC == "fp32" or cfg.TRAINER.BIOMEDDPT_ROBUST.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner.ctx"]

        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDDPT_ROBUST.PREC == "amp" else None
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMEDDPT_ROBUST.PREC
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

        loss_summary = {"loss": loss.item(), "acc": compute_accuracy(logits, label)[0].item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
