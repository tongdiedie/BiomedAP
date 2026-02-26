import os
from multiprocessing import Pool
import shutil
DATA = "data"
# NCTX = 4
CSC = False
CTP = "end"
CFG = "vit_b16"

# Robust 参数配置
LOW_TEMPLATE_TYPE = "minimal"  # minimal, article, empty, medical_minimal, generic

def run_experiment(args):
    method, model, dataset, shots, seed, gpu_id = args
    METHOD = method
    MODEL = model
    TRAINER = f"{METHOD}_{MODEL}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 

    # dir_path = f"output/{dataset}/shots_{shots}/{TRAINER}/nctx{NCTX}_csc{CSC}_ctp{CTP}_low{LOW_TEMPLATE_TYPE}/seed{seed}"
    dir_path = f"output/{dataset}/shots_{shots}/{TRAINER}/csc{CSC}_ctp{CTP}_low{LOW_TEMPLATE_TYPE}/seed{seed}"
    if METHOD == "CoOp" or METHOD == "CoCoOp" or METHOD == "VPT" or METHOD == "DPT" or METHOD == "Maple":
        config_yaml = f'configs/trainers/{METHOD}/{CFG}.yaml'
    else:
        config_yaml = f'configs/trainers/{METHOD}/few_shot/{dataset}.yaml'
    if False:
        print(f"Skipping existing job: {dir_path}")
    else:
        os.makedirs(dir_path, exist_ok=True) 
        print(f"Starting experiment: dataset={dataset}, shots={shots}, seed={seed}, gpu={gpu_id}")
        os.system(
            f"python train.py "
            f"--root {DATA} "
            f"--seed {seed} "
            f"--trainer {TRAINER} "
            f"--dataset-config-file configs/datasets/{dataset}.yaml "
            f"--config-file {config_yaml} "
            f"--output-dir {dir_path} "
            # f"TRAINER.{method.upper()}.N_CTX {NCTX} "
            f"TRAINER.{method.upper()}.CSC {CSC} "
            f"TRAINER.{method.upper()}.CLASS_TOKEN_POSITION {CTP} "
            f"TRAINER.{method.upper()}.LOW_TEMPLATE_TYPE {LOW_TEMPLATE_TYPE} "
            f"DATASET.NUM_SHOTS {shots} "
        )
        pro_path = os.path.join(dir_path, 'prompt_learner')
        ten_path = os.path.join(dir_path, 'tensorboard')
        if os.path.exists(pro_path):
            print(f"Deleting directory: {pro_path}")
            shutil.rmtree(pro_path)
        if os.path.exists(ten_path):
            print(f"Deleting directory: {ten_path}")
            shutil.rmtree(ten_path)

if __name__ == "__main__":
    datasets = ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]
    # datasets = ["covid", "dermamnist", "kneexray"]
    shots = [1, 2, 4, 8, 16]
    # shots = [0]
    # shots = [1]

    seeds = [1, 2, 3]
    gpu_ids = [0]  # 假设有 3 块 GPU，可调整
    # methods = ["BiomedAP", "BiomedDPT_Robust", "BiomedDPT", "BiomedCoOp", "KgCoOp", "CoOp", "CoCoOp", "ProGrad"]
    # methods = ["CoOp", "Maple", "VPT", "DPT"]
    methods = ["BiomedAP"]
    # models = ["BiomedCLIP", "CLIP", "PubMedCLIP", "PMCCLIP"]
    # models = ["BiomedCLIP"]
    models = ["BiomedCLIP"]
    # 生成任务列表，并循环分配 GPU
    tasks = [
        (method, model, dataset, shot, seed, gpu_ids[(i + j + k + l + h) % len(gpu_ids)])  # 轮流分配 GPU
        for h, model in enumerate(models)
        for k, method in enumerate(methods)
        for i, dataset in enumerate(datasets)
        for j, shot in enumerate(shots)
        for l, seed in enumerate(seeds)
    ]

    with Pool(processes=8) as pool:  # 并行 6 个进程 15
        pool.map(run_experiment, tasks)