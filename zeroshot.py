import os
import multiprocessing
import subprocess
import shutil
CFG = "vit_b16"
DATA = "data"
MODEL = "BiomedCLIP"
METHOD = "Zeroshot"



def run_experiment(args):
    model, dataset, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    TRAINER = f"Zeroshot{model}"
    
    DIR = f"output/{dataset}/{TRAINER}/{CFG}"
    if os.path.exists(DIR):
        print(f"Oops! The results exist at {DIR} (so skip this job)")
    else:
        subprocess.run([
            "python", "train.py",
            "--root", DATA,
            "--trainer", TRAINER,
            "--dataset-config-file", f"configs/datasets/{dataset}.yaml",
            "--config-file", f"configs/trainers/{METHOD}/{CFG}.yaml",
            "--output-dir", DIR,
            "--eval-only"
        ])
        pro_path = os.path.join(DIR, 'prompt_learner')
        ten_path = os.path.join(DIR, 'tensorboard')
        if os.path.exists(pro_path):
            print(f"Deleting directory: {pro_path}")
            shutil.rmtree(pro_path)
        if os.path.exists(ten_path):
            print(f"Deleting directory: {ten_path}")
            shutil.rmtree(ten_path)

if __name__ == "__main__":
    datasets = ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]
    gpu_ids = [0, 1, 2] 
    # models = ["BiomedCLIP", "CLIP", "PMCCLIP", "PubMedCLIP"] # zeroshot
    models = ["BiomedCLIP2", "CLIP2", "PMCCLIP2", "PubMedCLIP2"] # prompt+ensemble
    tasks = [
        (model, dataset, gpu_ids[(i + j) % len(gpu_ids)])  # 轮流分配 GPU
        for i, dataset in enumerate(datasets)
        for j, model in enumerate(models)
    ]


    with multiprocessing.Pool(processes=21) as pool:
        pool.map(run_experiment, tasks)