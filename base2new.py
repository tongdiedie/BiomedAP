import os
import subprocess
from multiprocessing import Pool
import os
import shutil

# 定义常量
DATA = "data"
MODEL = "BiomedCLIP"
CFG = "vit_b16_base2new"

LOADEP = 50
CTP = "end"
CSC = False
# NCTX = 4
SUB_base = "base"
SUB_novel = "new"

LOW_TEMPLATE_TYPE = "generic"  # minimal/article/generic/medical_minimal/empty


# 定义训练和测试函数
def run_experiment(args):
    method, dataset, seed, gpu_id = args
    shots = 16
    TRAINER = f"{method}_{MODEL}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
    if method == "CoOp" or method == "CoCoOp" or method == "KgCoOp" or method == "ProGrad":
        config_yaml = f'configs/trainers/{method}/vit_b16_base2new.yaml'
    else:
        config_yaml = f'configs/trainers/{method}/base_to_novel/{dataset}.yaml'
    if method == "CoCoOp":
        LOADEP = 10
    else:
        LOADEP = 50
    # 训练阶段
    # DIR = f"output/base2new/train_{SUB_base}/{dataset}/shots_{shots}/{TRAINER}/nctx{NCTX}_csc{CSC}_ctp{CTP}_low{LOW_TEMPLATE_TYPE}/seed{seed}"
    DIR = f"output/base2new/train_{SUB_base}/{dataset}/shots_{shots}/{TRAINER}/csc{CSC}_ctp{CTP}_low{LOW_TEMPLATE_TYPE}/seed{seed}"
    if os.path.exists(DIR):
        print(f"Oops! The results exist at {DIR} (so skip this job)")
    else:
        subprocess.run([
            "python", "train.py",
            "--root", DATA,
            "--seed", str(seed),
            "--trainer", TRAINER,
            "--dataset-config-file", f"configs/datasets/{dataset}.yaml",
            "--config-file", f"{config_yaml}",
            "--output-dir", DIR,
            "DATASET.NUM_SHOTS", str(shots),
            "DATASET.SUBSAMPLE_CLASSES", SUB_base,
            f"TRAINER.{method.upper()}.LOW_TEMPLATE_TYPE", LOW_TEMPLATE_TYPE
        ])
        

    # 测试阶段
    # COMMON_DIR = f"{dataset}/shots_{shots}/{TRAINER}/nctx{NCTX}_csc{CSC}_ctp{CTP}_low{LOW_TEMPLATE_TYPE}/seed{seed}"
    COMMON_DIR = f"{dataset}/shots_{shots}/{TRAINER}/csc{CSC}_ctp{CTP}_low{LOW_TEMPLATE_TYPE}/seed{seed}"
    MODEL_DIR = f"output/base2new/train_{SUB_base}/{COMMON_DIR}"
    DIR = f"output/base2new/test_{SUB_novel}/{COMMON_DIR}"
    if os.path.exists(DIR):
        print(f"Oops! The results exist at {DIR} (so skip this job)")
    else:
        subprocess.run([
            "python", "train.py",
            "--root", DATA,
            "--seed", str(seed),
            "--trainer", TRAINER,
            "--dataset-config-file", f"configs/datasets/{dataset}.yaml",
            "--config-file", f"{config_yaml}",
            "--output-dir", DIR,
            "--model-dir", MODEL_DIR,
            "--load-epoch", str(LOADEP),
            "--eval-only",
            "DATASET.NUM_SHOTS", str(shots),
            "DATASET.SUBSAMPLE_CLASSES", SUB_novel,
            f"TRAINER.{method.upper()}.LOW_TEMPLATE_TYPE", LOW_TEMPLATE_TYPE
        ])
        pro_path = os.path.join(MODEL_DIR, 'prompt_learner')
        ten_path = os.path.join(MODEL_DIR, 'tensorboard')
        if os.path.exists(pro_path):
            print(f"Deleting directory: {pro_path}")
            shutil.rmtree(pro_path)
        if os.path.exists(ten_path):
            print(f"Deleting directory: {ten_path}")
            shutil.rmtree(ten_path)

if __name__ == "__main__":

    # methods = ["BiomedCoOp", "KgCoOp", "CoOp", "CoCoOp", "ProGrad"]
    # datasets = ["dermamnist", "kneexray", "kvasir", "octmnist"]
    datasets = ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]
    # methods = ["BiomedDPT", "BiomedCoOp", "KgCoOp", "CoOp", "CoCoOp", "ProGrad"]
    # methods = ["BiomedDPT"]
    methods = ["BiomedAP"]
    # datasets = ["btmri"]

    seeds = [1, 2, 3]
    gpu_ids = [0]  # 假设有 3 块 GPU，可调整

    # 生成任务列表，并循环分配 GPU
    tasks = [
        (method, dataset, seed, gpu_ids[(i + k + j) % len(gpu_ids)])  # 轮流分配 GPU
        for k, method in enumerate(methods)
        for i, dataset in enumerate(datasets)
        for j, seed in enumerate(seeds)
    ]
    # 使用多进程并行运行任务
    with Pool(processes=6) as pool:
        pool.map(run_experiment, tasks)