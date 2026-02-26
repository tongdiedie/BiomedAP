import os
import argparse
import torch
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

datasets = ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]

biomedclip_model, _ = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
biomedclip_model = biomedclip_model.float().eval()
tokenizer = get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=4, type=int, help="Select top-k similar words")
args = parser.parse_args()
topk = args.topk

for dataset in datasets:
    
    fpath = f"output/{dataset}/shots_16/BiomedDPT_BiomedCLIP/cscFalse_ctpend_ctxinit/seed1/prompt_learner/model.pth.tar-100"
    
    if not os.path.exists(fpath):
        print(f"模型路径不存在: {fpath}")
        continue

    print(f"返回前 {topk} 个相似词")
    print(f"-------------------------------{dataset}-------------------------------")

    # 加载学习到的提示
    prompt_learner = torch.load(fpath, map_location="cpu", weights_only=False)["state_dict"]
    ctx = prompt_learner["prompt_learner.ctx"].float()  # 提取上下文向量并转换为 float

    # 获取词嵌入
    token_embedding = biomedclip_model.text.transformer.embeddings.word_embeddings.weight

    if ctx.dim() == 2:
        # 通用上下文
        distance = torch.cdist(ctx, token_embedding)  # 计算距离
        sorted_idxs = torch.argsort(distance, dim=1)[:, :topk]  # 获取前 topk 个词的索引
        output = ""
        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.tokenizer.decode(idx.item()) for idx in idxs]  # 解码词 ID
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]  # 获取距离
            output += f"{words[0]} ({dist[0]})\n"  # 格式化输出
        print(output)

    else:
        raise ValueError(f"不支持的 ctx 维度: {ctx.dim()}")