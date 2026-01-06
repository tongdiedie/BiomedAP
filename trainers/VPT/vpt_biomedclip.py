import os.path as osp
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os.path as osp
import warnings
warnings.filterwarnings("ignore")

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.prompt_templates import CUSTOM_TEMPLATES


# ================================================================= #
# 这个是VPT DEEP+CoOp（Frozen）+CLIP（Frozen）的实现。                #
# ================================================================= #


# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  VPT DEEP!                   #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class VPTDeepPromptLearner(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        # hyper param
        self.n_ctx = 4
        self.dtype = biomedclip_model.text.transformer.dtype
        self.ctx_dim = 768
        
        ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self):
        return self.ctx
    
class CLIP_Inplanted(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.image_encoder = biomedclip_model.visual
        self.dtype = biomedclip_model.text.transformer.dtype
        self.ctx_learner = VPTDeepPromptLearner(biomedclip_model)
        self.n_ctx = 4
            
    def forward(self, x):
        x = self.image_encoder.trunk.patch_embed(x)
        x = self.image_encoder.trunk._pos_embed(x)
        x = self.image_encoder.trunk.patch_drop(x)
        x = self.image_encoder.trunk.norm_pre(x)
        B = x.shape[0]
        ctx = self.ctx_learner()
        x = torch.cat((
            x[:, :1, :],
            ctx.expand(B, -1, -1).cuda(),
            x[:, 1+self.n_ctx:, :]
        ), dim=1)  
        for i in range(12):
            B = x.shape[0]            
            
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
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])
        
        with torch.no_grad():
            text_features = biomedclip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features.cuda()
        self.image_encoder = CLIP_Inplanted(biomedclip_model)
        # visual end
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        return logits

# end
@TRAINER_REGISTRY.register()
class VPT_BiomedCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedbiomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg,classnames, biomedbiomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "image_encoder.ctx_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
 
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.image_encoder.ctx_learner, cfg.MODEL.INIT_WEIGHTS)
            
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.image_encoder.ctx_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.image_encoder.ctx_learner, self.optim, self.sched)


        self.scaler = GradScaler() if cfg.TRAINER.VPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.VPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

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

        # By default, the best model is loaded
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

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)