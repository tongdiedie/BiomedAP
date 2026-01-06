import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets.busi
import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.kneexray
import datasets.dermamnist 
import datasets.octmnist
import datasets.oxfordpets

import trainers.Zeroshot.zeroshot
import trainers.CoOp.coop_clip
import trainers.CoOp.coop_biomedclip
import trainers.CoOp.coop_pubmedclip
import trainers.CoOp.coop_pmcclip
import trainers.CoCoOp.cocoop_clip
import trainers.CoCoOp.cocoop_biomedclip
import trainers.CoCoOp.cocoop_pubmedclip
import trainers.CoCoOp.cocoop_pmcclip
import trainers.KgCoOp.kgcoop_clip
import trainers.KgCoOp.kgcoop_biomedclip
import trainers.KgCoOp.kgcoop_pubmedclip
import trainers.KgCoOp.kgcoop_pmcclip
import trainers.ProGrad.prograd_clip
import trainers.ProGrad.prograd_biomedclip
import trainers.ProGrad.prograd_pubmedclip
import trainers.ProGrad.prograd_pmcclip
import trainers.BiomedCoOp.biomedcoop_clip
import trainers.BiomedCoOp.biomedcoop_biomedclip
import trainers.BiomedCoOp.biomedcoop_pubmedclip
import trainers.BiomedCoOp.biomedcoop_pmcclip
import trainers.BiomedDPT.biomeddpt_clip
import trainers.BiomedDPT.biomeddpt_biomedclip
import trainers.BiomedDPT.biomeddpt_pubmedclip
import trainers.BiomedDPT.biomeddpt_pmcclip
import trainers.VPT.vpt_biomedclip
import trainers.VPT.vpt_clip
import trainers.DPT.dpt_biomedclip
import trainers.DPT.dpt_clip
import trainers.MAPLE.maple_biomedclip
import trainers.MAPLE.maple_clip
import trainers.BiomedDPT_Robust.biomeddpt_robust_biomedclip
import trainers.BiomedDPT_Robust.biomeddpt_robust_clip
import trainers.BiomedDPT_Robust.biomeddpt_robust_pubmedclip
import trainers.BiomedDPT_Robust.biomeddpt_robust_pmcclip
import trainers.BiomedAP.biomedap_biomedclip
import trainers.BiomedAP.biomedap_clip
import trainers.BiomedAP.biomedap_pubmedclip
import trainers.BiomedAP.biomedap_pmcclip

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # class-specific context
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50

    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.KGCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.W = 1.0

    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROGRAD.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROGRAD.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.PROGRAD.GM = False
    cfg.TRAINER.PROGRAD.NAME = ""
    cfg.TRAINER.PROGRAD.ALPHA = 0.
    cfg.TRAINER.PROGRAD.T = 1.
    cfg.TRAINER.PROGRAD.LAMBDA = 1.
    
    cfg.TRAINER.BIOMEDDPT= CN()
    cfg.TRAINER.BIOMEDDPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BIOMEDDPT.CSC = False  # class-specific context
    cfg.TRAINER.BIOMEDDPT.CLASS_TOKEN_POSITION = "middle"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDDPT.N_CTX = 4  # number of context vectors
    cfg.TRAINER.BIOMEDDPT.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.BIOMEDDPT.L1_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDDPT.KL_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDDPT.N_PROMPTS = 50

    # ========== 【关键】添加 BiomedDPT_Robust 配置 ==========
    cfg.TRAINER.BIOMEDDPT_ROBUST = CN()
    cfg.TRAINER.BIOMEDDPT_ROBUST.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIOMEDDPT_ROBUST.CSC = False
    cfg.TRAINER.BIOMEDDPT_ROBUST.CLASS_TOKEN_POSITION = "middle" # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDDPT_ROBUST.N_CTX = 4
    cfg.TRAINER.BIOMEDDPT_ROBUST.PREC = "fp32"
    cfg.TRAINER.BIOMEDDPT_ROBUST.N_PROMPTS = 50
    # 【关键】低质量 Prompt 约束参数
    cfg.TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH = 12.5  # λ1：向高质量对齐
    cfg.TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA = 0.25       # λ2：知识蒸馏
    cfg.TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW = 0.3    # λ3：向低质量对齐【新增】
    cfg.TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE = "minimal"  # 低质量模板类型【新增】

    # ========== 【新增】BiomedContrast 配置 ==========
    cfg.TRAINER.BIOMEDCONTRAST = CN()
    # 基础参数
    cfg.TRAINER.BIOMEDCONTRAST.CTX_INIT = "a photo of a"  # 上下文初始化
    cfg.TRAINER.BIOMEDCONTRAST.CSC = False  # 是否使用class-specific context
    cfg.TRAINER.BIOMEDCONTRAST.CLASS_TOKEN_POSITION = "end"  # 类别token位置
    cfg.TRAINER.BIOMEDCONTRAST.N_CTX = 4  # 上下文向量数量
    cfg.TRAINER.BIOMEDCONTRAST.PREC = "fp32"  # 精度：fp16, fp32, amp
    cfg.TRAINER.BIOMEDCONTRAST.N_PROMPTS = 50  # BiomedDPT模板数量

    # ========== 【新增】BiomedAP 配置 ==========
    cfg.TRAINER.BIOMEDAP = CN()
    cfg.TRAINER.BIOMEDAP.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIOMEDAP.CSC = False
    cfg.TRAINER.BIOMEDAP.CLASS_TOKEN_POSITION = "middle"
    cfg.TRAINER.BIOMEDAP.N_CTX = 4
    cfg.TRAINER.BIOMEDAP.PREC = "fp32"
    cfg.TRAINER.BIOMEDAP.N_PROMPTS = 50
    cfg.TRAINER.BIOMEDAP.L1_LAMBDA_HIGH = 12.5
    cfg.TRAINER.BIOMEDAP.KL_LAMBDA = 0.25
    cfg.TRAINER.BIOMEDAP.L1_LAMBDA_LOW = 0.3
    cfg.TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE = "minimal"

    cfg.TRAINER.BIOMEDAP.ENABLE_FUSION = False      # 是否启用跨模态融合
    cfg.TRAINER.BIOMEDAP.FUSION_LAYERS = [5, 8]     # 融合层索引
    cfg.TRAINER.BIOMEDAP.ALIGNMENT_LAMBDA = 0.0     # Prompt对齐损失权重

    # 损失权重参数
    cfg.TRAINER.BIOMEDCONTRAST.L1_LAMBDA = 12.5  # λ1: L1损失权重（与正样本对齐）
    cfg.TRAINER.BIOMEDCONTRAST.KL_LAMBDA = 0.25  # λ2: KL散度权重（与zero-shot对齐）
    cfg.TRAINER.BIOMEDCONTRAST.REPULSION_LAMBDA = 0.1  # λ3: 负样本排斥损失权重【核心新增】
    cfg.TRAINER.BIOMEDCONTRAST.MARGIN = 0.3  # Margin值：控制排斥程度【核心新增】
    
    cfg.TRAINER.VPT= CN()
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.CSC = False  # class-specific context
    cfg.TRAINER.VPT.CLASS_TOKEN_POSITION = "middle"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.VPT.N_CTX = 4  # number of context vectors
    cfg.TRAINER.VPT.PREC = "fp32"  # fp16, fp32, amp
    
    cfg.TRAINER.DPT= CN()
    cfg.TRAINER.DPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.DPT.CSC = False  # class-specific context
    cfg.TRAINER.DPT.CLASS_TOKEN_POSITION = "middle"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.DPT.N_CTX = 4  # number of context vectors
    cfg.TRAINER.DPT.PREC = "fp32"  # fp16, fp32, amp
    
    cfg.TRAINER.MAPLE= CN()
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.CSC = False  # class-specific context
    cfg.TRAINER.MAPLE.CLASS_TOKEN_POSITION = "middle"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.MAPLE.N_CTX = 4  # number of context vectors
    cfg.TRAINER.MAPLE.PREC = "fp32"  # fp16, fp32, amp

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/DPT/vit_b16.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/btmri.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="DPT_BiomedCLIP", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", default=False,help="evaluation only") # True
    parser.add_argument(
        "--model-dir",
        type=str,
        default="output",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)