import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import math
import os
import requests
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import CUSTOM_BIOMEDDPT_TEMPLATES

from clip.pmcclip import ModifiedResNet
from transformers import AutoTokenizer, AutoModel

# PMC-CLIP模型文件配置
directory = "clip/checkpoints"
files = {
    "text_encoder.pth": "clip/checkpoints/text_encoder.pth",
    "image_encoder(resnet50).pth": "clip/checkpoints/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "clip/checkpoints/text_projection_layer.pth",
}

def download_file(url, filepath):
    """下载模型文件"""
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
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts, tokenized_prompts):
        output = self.text_encoder(
            inputs_embeds=prompts.cuda(), 
            attention_mask=tokenized_prompts['attention_mask'].cuda()
        )
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        return text_feature


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDDPT.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDDPT.CTX_INIT
        dtype = torch.float32
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            'clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract'
        )
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化context vectors
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(
                ctx_init, 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors='pt'
            )['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
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
        name_lens = [
            len(self.tokenizer(
                name, 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors='pt'
            )['input_ids']) 
            for name in classnames
        ]
        
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        tokenized_prompts = self.tokenizer(
            prompts, 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors='pt'
        )

        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(
                tokenized_prompts['input_ids'].cuda()
            ).type(dtype)

            # ===== 1. 预计算正样本特征 =====
            print("Computing fixed positive embeddings...")
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDDPT.N_PROMPTS

            for i in range(num_temp):
                prompt_texts = [BIOMEDDPT_TEMPLATES[classname][i] for classname in classnames]
                x_tokenized = self.tokenizer(
                    prompt_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=77,
                    return_tensors='pt'
                )
                x_tokenized_ids = x_tokenized['input_ids'].cuda()
                x_tokenized_attn = x_tokenized['attention_mask'].cuda()
                
                text_features = pmcclip_model.text_encoder(x_tokenized_ids, x_tokenized_attn)
                pooler_output = text_features.pooler_output
                text_features = pooler_output @ pmcclip_model.text_projection_layer
                all_teacher_features.append(text_features.unsqueeze(1))

            # ===== 2. 生成对比学习的负样本特征 =====
            print("Generating negative samples for contrastive learning...")
            print("  Strategy: Replacing class names in original templates")
            all_negative_features = []
            
            for target_idx in range(n_cls):
                target_class = classnames[target_idx]
                negative_features_per_class = []
                
                for neg_idx in range(n_cls):
                    if neg_idx == target_idx:
                        continue
                    
                    neg_class = classnames[neg_idx]
                    replaced_prompts = []
                    
                    # 用负类别名替换目标类的描述
                    for template_idx in range(num_temp):
                        original_prompt = BIOMEDDPT_TEMPLATES[target_class][template_idx]
                        replaced_prompt = original_prompt.replace(target_class, neg_class)
                        replaced_prompts.append(replaced_prompt)
                    
                    # 对替换后的prompts进行编码
                    replaced_tokenized = self.tokenizer(
                        replaced_prompts,
                        padding='max_length',
                        truncation=True,
                        max_length=77,
                        return_tensors='pt'
                    )
                    replaced_ids = replaced_tokenized['input_ids'].cuda()
                    replaced_attn = replaced_tokenized['attention_mask'].cuda()
                    
                    replaced_output = pmcclip_model.text_encoder(replaced_ids, replaced_attn)
                    replaced_pooler = replaced_output.pooler_output
                    replaced_features = replaced_pooler @ pmcclip_model.text_projection_layer
                    
                    # 对同一个负类的多个模板取平均
                    replaced_features = replaced_features.mean(dim=0, keepdim=True)
                    negative_features_per_class.append(replaced_features)
                
                # 堆叠该类的所有负样本特征 (n_cls-1, dim)
                negative_features_per_class = torch.cat(negative_features_per_class, dim=0)
                all_negative_features.append(negative_features_per_class.unsqueeze(0))
            
            # all_negative_features shape: (n_cls, n_cls-1, dim)
            self.fixed_negative_embeddings = torch.cat(all_negative_features, dim=0)
            print(f"✓ Generated negative embeddings: {self.fixed_negative_embeddings.shape}")
            print(f"  Example: '{classnames[0]}' -> '{classnames[1]}' in {classnames[0]}'s description")

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

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


class PMCCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained(
            'clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract'
        )

    def forward(self, image, text):
        encoded_input = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']

        return image_feature, text_feature


class CLIP_Inplanted(nn.Module):
    """带Visual Prompt的PMC-CLIP图像编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.image_encoder
        self.dtype = torch.float32
        self.num_tokens = 4
        
        # 针对ResNet50的不同层级定义prompt
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, 56, 56))
        self.deep_prompt_embeddings_1 = nn.Parameter(torch.zeros(1, self.num_tokens, 56, 56))
        self.deep_prompt_embeddings_2 = nn.Parameter(torch.zeros(1, self.num_tokens, 28, 28))
        self.deep_prompt_embeddings_3 = nn.Parameter(torch.zeros(1, self.num_tokens, 14, 14))
        self.deep_prompt_embeddings_4 = nn.Parameter(torch.zeros(1, self.num_tokens, 7, 7))
        self.prompt_dropout = nn.Dropout(0.5)

    def forward(self, x):
        # ResNet50的前向传播
        x = self.image_encoder.relu1(self.image_encoder.bn1(self.image_encoder.conv1(x)))
        x = self.image_encoder.relu2(self.image_encoder.bn2(self.image_encoder.conv2(x)))
        x = self.image_encoder.relu3(self.image_encoder.bn3(self.image_encoder.conv3(x)))
        x = self.image_encoder.avgpool(x)
        
        # 注入浅层prompt
        B = x.shape[0]
        x = torch.cat((
            x[:, :self.num_tokens, :, :],
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1, -1)),
            x[:, self.num_tokens:, :, :]
        ), dim=1)
        
        # Layer 1 + prompt
        x = self.image_encoder.layer1(x)
        x = torch.cat((
            x[:, :self.num_tokens, :, :],
            self.prompt_dropout(self.deep_prompt_embeddings_1.expand(B, -1, -1, -1)),
            x[:, self.num_tokens:, :, :]
        ), dim=1)
        
        # Layer 2 + prompt
        x = self.image_encoder.layer2(x)
        x = torch.cat((
            x[:, :self.num_tokens, :, :],
            self.prompt_dropout(self.deep_prompt_embeddings_2.expand(B, -1, -1, -1)),
            x[:, self.num_tokens:, :, :]
        ), dim=1)
        
        # Layer 3 + prompt
        x = self.image_encoder.layer3(x)
        x = torch.cat((
            x[:, :self.num_tokens, :, :],
            self.prompt_dropout(self.deep_prompt_embeddings_3.expand(B, -1, -1, -1)),
            x[:, self.num_tokens:, :, :]
        ), dim=1)
        
        # Layer 4 + prompt
        x = self.image_encoder.layer4(x)
        x = torch.cat((
            x[:, :self.num_tokens, :, :],
            self.prompt_dropout(self.deep_prompt_embeddings_4.expand(B, -1, -1, -1)),
            x[:, self.num_tokens:, :, :]
        ), dim=1)
        
        # Attention pooling
        x = self.image_encoder.attnpool(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, pmcclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = CLIP_Inplanted(pmcclip_model)
        self.text_encoder = TextEncoder(pmcclip_model)
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # 从配置读取参数
        self.margin = cfg.TRAINER.BIOMEDDPT.MARGIN
        self.repulsion_lambda = cfg.TRAINER.BIOMEDDPT.REPULSION_LAMBDA

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = math.exp(self.logit_scale)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 处理固定embeddings
        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:
            # Loss 1: 交叉熵
            loss_ce = F.cross_entropy(logits, label)
            
            # Loss 2: L1损失
            loss_l1 = F.l1_loss(text_features, fixed_embeddings.cuda(), reduction='mean') * \
                      self.cfg.TRAINER.BIOMEDDPT.L1_LAMBDA
            
            # Loss 3: KL散度
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel() * self.cfg.TRAINER.BIOMEDDPT.KL_LAMBDA

            # Loss 4: 对比学习负样本排斥损失
            fixed_negative_embeddings = self.prompt_learner.fixed_negative_embeddings.cuda()
            fixed_negative_embeddings = fixed_negative_embeddings / \
                                       fixed_negative_embeddings.norm(dim=-1, keepdim=True)
            
            batch_size = label.size(0)
            loss_repulsion = 0.0
            
            for i in range(batch_size):
                true_class = label[i].item()
                current_text_feature = text_features[true_class:true_class+1, :]
                negative_features = fixed_negative_embeddings[true_class, :, :]
                similarities = current_text_feature @ negative_features.t()
                repulsion_loss = torch.clamp(similarities + self.margin, min=0.0).mean()
                loss_repulsion += repulsion_loss
            
            loss_repulsion = (loss_repulsion / batch_size) * self.repulsion_lambda
            
            total_loss = loss_ce + loss_l1 + loss_kl + loss_repulsion
            
            return logits, total_loss, {
                'loss_ce': loss_ce.item(),
                'loss_l1': loss_l1.item(),
                'loss_kl': loss_kl.item(),
                'loss_repulsion': loss_repulsion.item()
            }
        else:
            return logits


@TRAINER_REGISTRY.register()
class BiomedDPT_Contrast_PMCCLIP(TrainerX):
    """
    BiomedDPT with Contrastive Learning on PMC-CLIP backbone
    
    特点:
    - 使用PMC-CLIP (ResNet50) 作为backbone
    - 负样本排斥策略: 通过替换类别名称生成负样本
    - Visual Prompt: 在ResNet的不同层级注入可学习的prompt
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDDPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # 检查并下载PMC-CLIP模型文件
        for filename, url in files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PMC-CLIP (backbone: ResNet50)")
        
        # 加载图像编码器
        image_encoder = ModifiedResNet(
            layers=[3, 4, 6, 3], 
            output_dim=768, 
            heads=8, 
            image_size=224, 
            width=64
        )
        image_encoder.load_state_dict(
            torch.load(os.path.join(directory, 'image_encoder(resnet50).pth'), weights_only=True)
        )

        # 加载文本编码器
        text_encoder = AutoModel.from_pretrained(
            'clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract'
        )
        text_encoder.load_state_dict(
            torch.load(os.path.join(directory, 'text_encoder.pth'), weights_only=True)
        )

        # 加载文本投影层
        text_projection_layer = torch.load(
            os.path.join(directory, 'text_projection_layer.pth'), 
            weights_only=True
        )
        text_projection_layer = nn.Parameter(text_projection_layer)

        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()

        print("Building custom PMC-CLIP with contrastive learning")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model)

        print("Turning off gradients in image and text encoder")
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
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDDPT.PREC == "amp" else None
        
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

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }
        
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

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} from \"{}\" (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
