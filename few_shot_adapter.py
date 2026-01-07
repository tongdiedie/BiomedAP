import os
import subprocess
from multiprocessing import Pool

# 自定义配置
DATA = "data"
MODEL = "BiomedCLIP"
# METHOD = f"TIPAdapter_{MODEL}"



def run_experiment(args):
    method, dataset, shots, gpu_id = args
    dic = {
        "TIPAdapter": "TiP_Adapter",
        "LinearProbe": "LP",
        "LinearProbe_P2": "LP2",
        "ClipAdapter": "CLIP_Adapter"
    }
    METHOD = f"{method}_{MODEL}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
    dir_path = f"output/{dataset}/shots_{shots}/{METHOD}/"
    if os.path.exists(dir_path):
        print(f"Oops! The results exist at {dir_path} (so skip this job)")
        return

    os.makedirs(dir_path, exist_ok=True)

    command = [
        "python", "main.py",
        "--base_config", f"configs/trainers/{dic[method]}/few_shot.yaml",
        "--dataset_config", f"configs/datasets/{dataset}.yaml",
        "--opt", "root_path", DATA,
        "output_dir", dir_path,
        "shots", str(shots),
        "method", METHOD,
        "clip_model", MODEL
    ]

    subprocess.run(command)

def main():
    gpu_ids = [0, 1, 2]  
    datasets = ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]
    # datasets = ["btmri", "busi"]
    shots = [1, 2, 4, 8, 16]
    # methods = ["ClipAdapter","TIPAdapter", "LinearProbe","LinearProbe_P2"]
    methods = ["TIPAdapter"]
    tasks = [
        (method, dataset, shot, gpu_ids[(i + j + k) % len(gpu_ids)])  
        for k, method in enumerate(methods)
        for i, dataset in enumerate(datasets)
        for j, shot in enumerate(shots)
    ]


    with Pool(processes=6) as pool:
        pool.map(run_experiment, tasks)

if __name__ == "__main__":
    main()