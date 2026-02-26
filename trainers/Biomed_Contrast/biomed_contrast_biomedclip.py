# 总损失公式
# total_loss = loss_ce + loss_l1 + loss_kl + loss_repulsion * REPULSION_LAMBDA
from collections import OrderedDict
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

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        x = self.model.encode_text(prompts, True, tokenized_prompts)
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDDPT.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDDPT.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
                                       cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # 使用给定的词来初始化上下文向量
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化
            if cfg.TRAINER.BIOMEDDPT.CSC:
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

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])
        
        # 创建冻结的CLIP模型用于提取固定特征
        biomedclip_model_temp, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            
            # ===== 1. 预计算正样本特征（原有逻辑）=====
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDDPT.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i]) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))
            
            # ===== 2. 新增：通过替换类别名称生成负样本特征 =====
            print("Generating negative samples by replacing class names in prompts...")
            all_negative_features = []
            
            # 对于每个目标类别
            for target_idx in range(n_cls):
                target_class = classnames[target_idx]  # 当前目标类别名
                negative_features_per_class = []
                
                # 遍历所有其他类别作为替换源
                for neg_idx in range(n_cls):
                    if neg_idx == target_idx:  # 跳过自己
                        continue
                    
                    neg_class = classnames[neg_idx]  # 负样本类别名
                    
                    # 关键：用负类别名替换目标类的描述
                    # 例如: "A normal brain in MRI..." -> "A glioma tumor in MRI..."
                    # 使用BIOMEDDPT_TEMPLATES的多个模板
                    replaced_prompts = []
                    for template_idx in range(num_temp):
                        # 获取目标类的原始模板描述
                        original_prompt = BIOMEDDPT_TEMPLATES[target_class][template_idx]
                        # 将其中的类别名替换为负类别名
                        # 方法：简单的字符串替换
                        replaced_prompt = original_prompt.replace(target_class, neg_class)
                        replaced_prompts.append(replaced_prompt)
                    
                    # 对替换后的prompts进行编码
                    replaced_tokenized = torch.cat([self.tokenizer(p) for p in replaced_prompts])
                    replaced_features = biomedclip_model_temp.encode_text(replaced_tokenized.cuda())
                    
                    # 对同一个负类的多个模板取平均
                    replaced_features = replaced_features.mean(dim=0, keepdim=True)
                    negative_features_per_class.append(replaced_features)
                
                # 堆叠该类的所有负样本特征 (n_cls-1, dim)
                negative_features_per_class = torch.cat(negative_features_per_class, dim=0)
                all_negative_features.append(negative_features_per_class.unsqueeze(0))
            
            # all_negative_features shape: (n_cls, n_cls-1, dim)
            self.fixed_negative_embeddings = torch.cat(all_negative_features, dim=0)
            print(f"✓ Generated negative embeddings with shape: {self.fixed_negative_embeddings.shape}")
            print(f"  Strategy: Replaced class names in original templates")
            print(f"  Example: '{classnames[0]}' -> '{classnames[1]}' in {classnames[0]}'s description")

        # 正样本特征 shape: (n_cls, num_temp, dim)
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMEDDPT.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

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
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
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
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.text.transformer.dtype
        self.num_tokens = 4
        self.prompt_dim = 768
        self.prompt_embeddings = torch.zeros(1, self.num_tokens, self.prompt_dim)
        self.deep_prompt_embeddings = torch.zeros(12, self.num_tokens, self.prompt_dim)
        self.prompt_dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.image_encoder.trunk.patch_embed(x)
        x = self.image_encoder.trunk._pos_embed(x)
        x = self.image_encoder.trunk.patch_drop(x)
        x = self.image_encoder.trunk.norm_pre(x)
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1).cuda()),
            x[:, 1+self.num_tokens:, :]
        ), dim=1)  
        for i in range(12):
            B = x.shape[0]
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)  
            x = self.image_encoder.trunk.blocks[i](x)
        x = self.image_encoder.trunk.norm(x)
        x = x[:, 0]
        x = self.image_encoder.trunk.fc_norm(x)
        x = self.image_encoder.trunk.head_drop(x)
        x = self.image_encoder.trunk.head(x)
        x = self.image_encoder.head(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = CLIP_Inplanted(biomedclip_model)
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        self.margin = cfg.TRAINER.BIOMEDDPT.MARGIN
        self.repulsion_lambda = cfg.TRAINER.BIOMEDDPT.REPULSION_LAMBDA

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        # 计算prompted image和text特征
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 处理正样本固定embeddings
        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.mean(dim=1)  # 对多个模板取平均
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 计算zero-shot logits
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        
        # 计算学习的logits
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:         
            # ===== Loss 1: 交叉熵损失（基础分类损失）=====
            loss_ce = F.cross_entropy(logits, label)
            
            # ===== Loss 2: L1损失（与正样本描述对齐）=====
            loss_l1 = F.l1_loss(text_features, fixed_embeddings.cuda(), reduction='mean') * self.cfg.TRAINER.BIOMEDDPT.L1_LAMBDA 
            
            # ===== Loss 3: KL散度损失（与zero-shot预测对齐）=====
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel() * self.cfg.TRAINER.BIOMEDDPT.KL_LAMBDA

            # ===== Loss 4: 负样本排斥损失（新增：远离替换后的错误描述）=====
            # 核心思想：让学到的特征远离"用其他类名替换自己类名"的错误描述
            # 例如：normal brain的特征应该远离"A glioma tumor in MRI appears with..."这种描述
            
            # 获取负样本embeddings并归一化
            fixed_negative_embeddings = self.prompt_learner.fixed_negative_embeddings.cuda()
            fixed_negative_embeddings = fixed_negative_embeddings / fixed_negative_embeddings.norm(dim=-1, keepdim=True)
            
            batch_size = label.size(0)
            loss_repulsion = 0.0
            
            for i in range(batch_size):
                # 获取当前样本的真实类别
                true_class = label[i].item()
                
                # 获取该类别学到的文本特征 (1, dim)
                current_text_feature = text_features[true_class:true_class+1, :]
                
                # 获取该类别对应的负样本特征 (n_cls-1, dim)
                # 这些是用其他类名替换当前类名后的描述特征
                negative_features = fixed_negative_embeddings[true_class, :, :]
                
                # 计算与所有负样本的余弦相似度 (1, n_cls-1)
                similarities = current_text_feature @ negative_features.t()
                
                # 使用margin-based hinge loss
                # 目标：让相似度小于 -margin（即尽可能不相似）
                # loss = max(0, similarity + margin)
                repulsion_loss = torch.clamp(similarities + self.margin, min=0.0).mean()
                loss_repulsion += repulsion_loss
            
            loss_repulsion = loss_repulsion / batch_size
            
            # 获取排斥损失权重
            loss_repulsion = loss_repulsion * self.repulsion_lambda
            
            # ===== 总损失 =====
            total_loss = loss_ce + loss_l1 + loss_kl + loss_repulsion
            
            # 返回详细的损失字典用于监控
            return logits, total_loss, {
                'loss_ce': loss_ce.item(),
                'loss_l1': loss_l1.item(),
                'loss_kl': loss_kl.item(),
                'loss_repulsion': loss_repulsion.item()
            }
        else:
            return logits


@TRAINER_REGISTRY.register()
class BiomedDPT_BiomedCLIP(TrainerX):
    """
    BiomedDPT with BiomedCLIP backbone
    
    新增负样本排斥策略：
    - 通过替换类别名称生成负样本描述
    - 例如："A normal brain..." -> "A glioma tumor..."
    - 训练时让特征远离这些错误的描述
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDDPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        if cfg.TRAINER.BIOMEDDPT.PREC == "fp32" or cfg.TRAINER.BIOMEDDPT.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP with negative sample repulsion")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner.ctx"]

        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)
        
        # 确认需要更新的参数
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # 只给prompt_learner优化器
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDDPT.PREC == "amp" else None
        
        # 多GPU训练支持
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMEDDPT.PREC
        if prec == "amp":
            with autocast():
                output = model(image, label)
                if len(output) == 3:
                    logits, loss, loss_dict = output
                else:
                    logits, loss = output
                    loss_dict = {}
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            output = model(image, label)
            if len(output) == 3:
                logits, loss, loss_dict = output
            else:
                logits, loss = output
                loss_dict = {}
            
            self.model_backward_and_update(loss)

        # 构建损失摘要
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }
        
        # 添加详细的损失项
        if loss_dict:
            loss_summary.update(loss_dict)

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

        # 默认加载最佳模型
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 忽略固定的token向量
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # 设置strict=False
            self._models[name].load_state_dict(state_dict, strict=False)