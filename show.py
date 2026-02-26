import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden
import pandas as pd
import os
import shutil
from tabulate import tabulate 
file_names = {
    'btmri': 'BTMRI.csv',
    'busi': 'BUSI.csv',
    'covid': 'COVID_19.csv',
    'ctkidney': 'CTKidney.csv',
    'dermamnist': 'DermaMNIST.csv',
    'kvasir': 'Kvasir.csv',
    'kather': 'Kather_texture.csv',
    'lungcolon': 'LungColon.csv',
    'retina': 'RETINA.csv',
    'kneexray': 'KneeXray.csv',
    'octmnist': 'OCTMNIST.csv',
    'chmnist': 'CHMNIST.csv'
}
metric = {
    "name": "accuracy",
    "regex": re.compile(fr"\* accuracy: ([\.\deE+-]+)%"),
}
datasets =  ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]
# cols = ["CoOp", "CoCoOp", "KgCoOp", "ProGrad", "MyCoOp" ]
# cols = ["ProGrad"]
cols = ["BiomedAP"]

shots = [1, 2, 4, 8, 16]
def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))
# def parse_function(*metrics, directory=""):
    
#     subdirs = listdir_nohidden(directory, sort=True)
#     outputs = []
#     elapsed_time = 0
#     for subdir in subdirs:
#         fpath = osp.join(directory, subdir, "log.txt")
#         if not os.path.exists(fpath):
#             shutil.rmtree(osp.join(directory, subdir))
#             continue
#         # assert check_isfile(fpath)
#         output = OrderedDict()

#         with open(fpath, "r") as f:
#             lines = f.readlines()

#             for line in lines:
#                 line = line.strip()

#                 for metric in metrics:
#                     match = metric["regex"].search(line)

#                     pattern = r"Elapsed: (\d{1,2}:\d{2}:\d{2})"

#                     match_time = re.search(pattern, line)

#                     if match_time:
#                         time_str = match_time.group(1)  # 提取捕获组中的时间
#                         h, m, s = map(int, time_str.split(':'))
#                         elapsed_time = h * 3600 + m * 60 + s
 
#                     if match:
#                         if "file" not in output:
#                             output["file"] = fpath
#                         num = float(match.group(1))
#                         name = metric["name"]
#                         output[name] = num

#         if output:
#             outputs.append(output)
#         else:
#             folder_path = os.path.dirname(fpath)
#             shutil.rmtree(folder_path)
#             print(f"文件夹 {folder_path} 已被删除")

#     metrics_results = defaultdict(list)

#     for output in outputs:
#         msg = ""
#         for key, value in output.items():
#             if isinstance(value, float):
#                 msg += f"{key}: {value:.2f}%. "
#             else:
#                 msg += f"{key}: {value}. "
#             if key != "file":
#                 metrics_results[key].append(value)

#     output_results = OrderedDict()

#     for key, values in metrics_results.items():
#         avg = np.mean(values)
#         output_results[key] = avg
#         std = compute_ci95(values)

#     return avg, std, elapsed_time
def parse_function(*metrics, directory=""):
    subdirs = listdir_nohidden(directory, sort=True)

    seed_to_metric = {}   # e.g. {"seed1": 78.1, "seed2": 77.9, ...}
    elapsed_time = 0

    metric_name = metrics[0]["name"]

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        if not os.path.exists(fpath):
            # 你如果不想删目录，可以注释掉下一行
            shutil.rmtree(osp.join(directory, subdir))
            continue

        found_value = None

        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()

                # time（如果一份log里多次出现Elapsed，你可能想取最后一次；这里沿用你原逻辑）
                match_time = re.search(r"Elapsed: (\d{1,2}:\d{2}:\d{2})", line)
                if match_time:
                    h, m, s = map(int, match_time.group(1).split(':'))
                    elapsed_time = h * 3600 + m * 60 + s

                # metric
                match = metrics[0]["regex"].search(line)
                if match:
                    found_value = float(match.group(1))

        if found_value is not None:
            seed_to_metric[subdir] = found_value
        else:
            folder_path = os.path.dirname(fpath)
            shutil.rmtree(folder_path)
            print(f"文件夹 {folder_path} 已被删除")

    values = list(seed_to_metric.values())
    if len(values) == 0:
        return 0.0, 0.0, 0, seed_to_metric

    avg = float(np.mean(values))
    std = float(compute_ci95(values))

    return avg, std, elapsed_time, seed_to_metric


def show_base_to_new():
    out_dir = "output_train_csv/base2new_medical_minimal"
    os.makedirs(out_dir, exist_ok=True)

    all_detail_rows = []  # 用来收集所有 dataset 的 dfs

    datasets =  ["btmri", "busi", "chmnist", "covid", "ctkidney", "dermamnist", "kneexray", "kvasir", "lungcolon", "octmnist", "retina"]
    base_data = {col: [] for col in cols}
    novel_data = {col: [] for col in cols}
    hm_data = {col: [] for col in cols}

    best_base_data = {col: [] for col in cols}
    best_novel_data = {col: [] for col in cols}
    best_hm_data = {col: [] for col in cols}
    
    # 用于存储所有数据集的表格
    all_tables = []
    
    for dataset in datasets:
        expected_seeds = ["seed1", "seed2", "seed3"]
        # data = {
        #     "Base": [],
        #     "Novel": [],
        # }
        data = {}
        for col in cols:
            base_directory = f"output/base2new/train_base/{dataset}/shots_16/{col}_BiomedCLIP/cscFalse_ctpend_lowmedical_minimal"
            novel_directory = f"output/base2new/test_new/{dataset}/shots_16/{col}_BiomedCLIP/cscFalse_ctpend_lowmedical_minimal"
            # acc, std, time = parse_function(metric, directory=base_directory)
            # data['Base'].append(acc.round(2))  
            # acc, std, time = parse_function(metric, directory=novel_directory)
            # data['Novel'].append(acc.round(2))
            acc_b, std_b, time_b, seedmap_b = parse_function(metric, directory=base_directory)
            acc_n, std_n, time_n, seedmap_n = parse_function(metric, directory=novel_directory)

            # 固定 seed 列：Base_seed1/Base_seed2/Base_seed3
            for sd in expected_seeds:
                data.setdefault(f"Base_{sd}", [])
                v = seedmap_b.get(sd, np.nan)
                data[f"Base_{sd}"].append(round(v, 2) if not np.isnan(v) else np.nan)

            for sd in expected_seeds:
                data.setdefault(f"Novel_{sd}", [])
                v = seedmap_n.get(sd, np.nan)
                data[f"Novel_{sd}"].append(round(v, 2) if not np.isnan(v) else np.nan)

            # 均值 + CI95
            data.setdefault("Base_mean", [])
            data.setdefault("Base_ci95", [])
            data["Base_mean"].append(round(acc_b, 2))
            data["Base_ci95"].append(round(std_b, 2))

            data.setdefault("Novel_mean", [])
            data.setdefault("Novel_ci95", [])
            data["Novel_mean"].append(round(acc_n, 2))
            data["Novel_ci95"].append(round(std_n, 2))

            
        dfs = pd.DataFrame(data)
        dfs.index = cols
        seed_cols = ["seed1", "seed2", "seed3"]
        # Base / Novel 的 best（各自取最大）
        dfs["Base_best"] = dfs[[f"Base_{s}" for s in seed_cols]].max(axis=1)
        dfs["Novel_best"] = dfs[[f"Novel_{s}" for s in seed_cols]].max(axis=1)

        # dfs['HM'] = dfs.iloc[:, :2].mean(axis=1).round(2)
        dfs["HM"] = dfs[["Base_mean", "Novel_mean"]].mean(axis=1).round(2)
        dfs["HM_seed1"] = dfs[["Base_seed1","Novel_seed1"]].mean(axis=1)
        dfs["HM_seed2"] = dfs[["Base_seed2","Novel_seed2"]].mean(axis=1)
        dfs["HM_seed3"] = dfs[["Base_seed3","Novel_seed3"]].mean(axis=1)

        
        dfs["HM_best"] = dfs[["HM_seed1", "HM_seed2", "HM_seed3"]].max(axis=1).round(2)
        dfs.index.name = dataset
        
        # 将当前数据集的表格转换为字符串并存储
        all_tables.append(tabulate(dfs, headers='keys', tablefmt='pretty'))
        print(tabulate(dfs, headers='keys', tablefmt='pretty'))

        # ===== 收集到总明细表（所有dataset合并）=====
        tmp = dfs.copy()
        tmp.index.name = "Method"
        tmp = tmp.reset_index()

        # Dataset 放在第一列
        tmp.insert(0, "Dataset", dataset)

        all_detail_rows.append(tmp)
        
        # dfs.to_csv(f"output_train_csv/{file_names[dataset]}")
        for col in cols:
            # base_data[col].append(dfs.loc[col, 'Base'])
            # novel_data[col].append(dfs.loc[col, 'Novel'])
            base_data[col].append(float(dfs.loc[col, "Base_mean"]))
            novel_data[col].append(float(dfs.loc[col, "Novel_mean"]))
            hm_data[col].append(float(dfs.loc[col, "HM"]))

            best_base_data[col].append(float(dfs.loc[col, "Base_best"]))
            best_novel_data[col].append(float(dfs.loc[col, "Novel_best"]))
            best_hm_data[col].append(float(dfs.loc[col, "HM_best"]))

    # ===== 合并所有dataset明细并输出一个大CSV ===== 
    if len(all_detail_rows) > 0:
        all_detail_df = pd.concat(all_detail_rows, ignore_index=True)
        all_detail_path = os.path.join(out_dir, "base2new_all_detail.csv")
        all_detail_df.to_csv(all_detail_path, index=False)
        print(f"\n已保存合并明细CSV: {all_detail_path}")
    else:
        print("\n警告：all_detail_rows 为空，没有生成合并明细CSV")
    base_avg = {col: round(sum(base_data[col]) / len(base_data[col]), 2) for col in cols}
    novel_avg = {col: round(sum(novel_data[col]) / len(novel_data[col]), 2) for col in cols}
    hm_avg = {col: round(sum(hm_data[col]) / len(hm_data[col]), 2) for col in cols}

    best_base_avg = {col: round(sum(best_base_data[col]) / len(best_base_data[col]), 2) for col in cols}
    best_novel_avg = {col: round(sum(best_novel_data[col]) / len(best_novel_data[col]), 2) for col in cols}
    best_hm_avg = {col: round(sum(best_hm_data[col]) / len(best_hm_data[col]), 2) for col in cols}

    best_avg_df = pd.DataFrame({
        "Method": cols,
        "Best Base Average": [best_base_avg[col] for col in cols],
        "Best Novel Average": [best_novel_avg[col] for col in cols],
        "Best HM Average": [best_hm_avg[col] for col in cols],
    }).set_index("Method")

    avg_df = pd.DataFrame({
        'Method': cols,
        'Base Average': [base_avg[col] for col in cols],
        'Novel Average': [novel_avg[col] for col in cols],
        'HM Average': [hm_avg[col] for col in cols],
    }).set_index('Method')

    best_avg_df = pd.DataFrame({
        "Method": cols,
        "Best Base Average": [best_base_avg[col] for col in cols],
        "Best Novel Average": [best_novel_avg[col] for col in cols],
        "Best HM Average": [best_hm_avg[col] for col in cols],
    }).set_index("Method")

    print("\n每个方法的 Base、Novel、HM平均值：")
    print(tabulate(avg_df, headers="keys", tablefmt="pretty"))

    print("\n每个方法的 best Base、best Novel、best HM 平均值：")
    print(tabulate(best_avg_df, headers="keys", tablefmt="pretty"))

    # ===== 输出两张汇总CSV =====
    avg_df.to_csv(os.path.join(out_dir, "base2new_mean_summary.csv"))
    best_avg_df.to_csv(os.path.join(out_dir, "base2new_best_summary.csv"))
    print(f"\n已保存汇总CSV: {os.path.join(out_dir, 'base2new_mean_summary.csv')}")
    print(f"已保存汇总CSV: {os.path.join(out_dir, 'base2new_best_summary.csv')}")



def show_three_tables(all_tables):
    for i in range(0, len(all_tables), 3):
        # 获取当前批次的三个表格
        batch_tables = all_tables[i:i+3]
        
        # 如果不足三个表格，用空表格补齐
        while len(batch_tables) < 3:
            batch_tables.append("")
        
        # 将每个表格的行拆分
        table1_lines = batch_tables[0].split('\n')
        table2_lines = batch_tables[1].split('\n')
        table3_lines = batch_tables[2].split('\n')
        
        # 确保每个表格的行数相同
        max_lines = max(len(table1_lines), len(table2_lines), len(table3_lines))
        table1_lines += [''] * (max_lines - len(table1_lines))
        table2_lines += [''] * (max_lines - len(table2_lines))
        table3_lines += [''] * (max_lines - len(table3_lines))
        
        # 将三个表格并排拼接
        combined_table = '\n'.join(
            f"{line1}    {line2}    {line3}"
            for line1, line2, line3 in zip(table1_lines, table2_lines, table3_lines)
        )
        
        # 输出当前批次的并排表格
        print(combined_table)
        print()  # 添加空行分隔不同批次的表格

def show_few_shot_coop():
    cols = ["DPT", "VPT", "Maple"]

    k1_data = {col: [] for col in cols}
    k2_data = {col: [] for col in cols}
    k4_data = {col: [] for col in cols}
    k8_data = {col: [] for col in cols}
    k16_data = {col: [] for col in cols}
    avg_data = {col: [] for col in cols}
    # ["BiomedCLIP", "CLIP", "PMCCLIP", "PubMedCLIP"]
    for dataset in datasets:
        data = {
            "k=1": [],
            "k=2": [],
            "k=4": [],
            "k=8": [],
            "k=16": [],
            "avg": [],
            "time": [],
        }
        for col in cols:
            all_time, all_avg = 0, 0
            for shot in shots:
                directory = f"output/{dataset}/shots_{shot}/{col}_BiomedCLIP/cscFalse_ctpend_lowminimal"
                acc, std, time = parse_function(metric, directory=directory)       
                # data[f'k={shot}'].append(f"{acc.round(2)}+- {std.round(2):.2f}%. ")  
                data[f'k={shot}'].append(acc.round(2))  
                all_time = all_time + time
                all_avg = all_avg + acc
            all_time = round(all_time / 5, 2)
            all_avg = round(all_avg / 5, 2)
            data['time'].append(all_time)
            data['avg'].append(all_avg)
        
        dfs = pd.DataFrame(data)
        dfs.index = cols
        dfs.index.name = dataset
        print(tabulate(dfs, headers='keys', tablefmt='pretty'))
        # ===== 收集到总明细表（每个dataset一行/每个method一行）=====
        tmp = dfs.copy()
        tmp.index.name = "Method"
        tmp = tmp.reset_index()

        # Dataset 放在第一列
        tmp.insert(0, "Dataset", dataset)

        all_detail_rows.append(tmp)

        for col in cols:
            k1_data[col].append(dfs.loc[col, 'k=1'])
            k2_data[col].append(dfs.loc[col, 'k=2'])
            k4_data[col].append(dfs.loc[col, 'k=4'])
            k8_data[col].append(dfs.loc[col, 'k=8'])
            k16_data[col].append(dfs.loc[col, 'k=16'])
            avg_data[col].append(dfs.loc[col, 'avg'])
        # dfs.to_csv(f"output_train_csv/{file_names[dataset]}")
    k1_avg = {col: round(sum(k1_data[col]) / len(k1_data[col]), 2) for col in cols}
    k2_avg = {col: round(sum(k2_data[col]) / len(k2_data[col]), 2) for col in cols}
    k4_avg = {col: round(sum(k4_data[col]) / len(k4_data[col]), 2) for col in cols}
    k8_avg = {col: round(sum(k8_data[col]) / len(k8_data[col]), 2) for col in cols}
    k16_avg = {col: round(sum(k16_data[col]) / len(k16_data[col]), 2) for col in cols}
    all_avg = {col: round(sum(avg_data[col]) / len(avg_data[col]), 2) for col in cols}
     # 将 HM 平均值转换为 DataFrame 以便输出
    avg_df = pd.DataFrame({
        'Method': cols,
        'k=1': [k1_avg[col] for col in cols],
        'k=2': [k2_avg[col] for col in cols],
        'k=4': [k4_avg[col] for col in cols],
        'k=8': [k8_avg[col] for col in cols],
        'k=16': [k16_avg[col] for col in cols],
        'Average': [all_avg[col] for col in cols],
    })
    avg_df.set_index('Method', inplace=True)
    # avg_df.to_csv(f"output_train_csv/avg.csv")
    # 打印 HM 平均值表格
    print("\n每个方法的平均值：")
    print(tabulate(avg_df, headers='keys', tablefmt='pretty'))

def show_few_shot_adapter():
    # cols = ["ClipAdapter","TIPAdapter", "LinearProbe","LinearProbe_P2"]
    cols = ['TIPAdapter']
    k1_data = {col: [] for col in cols}
    k2_data = {col: [] for col in cols}
    k4_data = {col: [] for col in cols}
    k8_data = {col: [] for col in cols}
    k16_data = {col: [] for col in cols}
    avg_data = {col: [] for col in cols}
    for dataset in datasets:
        data = {
            "k=1": [],
            "k=2": [],
            "k=4": [],
            "k=8": [],
            "k=16": [],
        }
        for col in cols: 
            for shot in shots:
                df_path = f"output/{dataset}/shots_{shot}/{col}_BiomedCLIP/{file_names[dataset]}"
                if os.path.exists(df_path):
                    df = pd.read_csv(df_path)
                    acc_values = df['acc'].values.round(2).tolist()  # 将数组转换为列表
                    data[f'k={shot}'].extend(acc_values) 
                else:
                    shutil.rmtree(os.path.dirname(df_path))
                    print(f"文件夹 {df_path} 已被删除")
                    continue
                    
        dfs = pd.DataFrame(data)
        dfs.index = cols
        dfs['avg'] = dfs.iloc[:, :5].mean(axis=1).round(2)
        dfs.index.name = dataset
        print(tabulate(dfs, headers='keys', tablefmt='pretty'))
        # dfs.to_csv(f"output_main_csv/{file_names[dataset]}")
        for col in cols:
            k1_data[col].append(dfs.loc[col, 'k=1'])
            k2_data[col].append(dfs.loc[col, 'k=2'])
            k4_data[col].append(dfs.loc[col, 'k=4'])
            k8_data[col].append(dfs.loc[col, 'k=8'])
            k16_data[col].append(dfs.loc[col, 'k=16'])
            avg_data[col].append(dfs.loc[col, 'avg'])
    k1_avg = {col: round(sum(k1_data[col]) / len(k1_data[col]), 2) for col in cols}
    k2_avg = {col: round(sum(k2_data[col]) / len(k2_data[col]), 2) for col in cols}
    k4_avg = {col: round(sum(k4_data[col]) / len(k4_data[col]), 2) for col in cols}
    k8_avg = {col: round(sum(k8_data[col]) / len(k8_data[col]), 2) for col in cols}
    k16_avg = {col: round(sum(k16_data[col]) / len(k16_data[col]), 2) for col in cols}
    all_avg = {col: round(sum(avg_data[col]) / len(avg_data[col]), 2) for col in cols}
    # 将 HM 平均值转换为 DataFrame 以便输出
    avg_df = pd.DataFrame({
        'Method': cols,
        'k=1': [k1_avg[col] for col in cols],
        'k=2': [k2_avg[col] for col in cols],
        'k=4': [k4_avg[col] for col in cols],
        'k=8': [k8_avg[col] for col in cols],
        'k=16': [k16_avg[col] for col in cols],
        'Average': [all_avg[col] for col in cols],
    })
    avg_df.set_index('Method', inplace=True)

    # 打印 HM 平均值表格
    print("\n每个方法的平均值：")
    print(tabulate(avg_df, headers='keys', tablefmt='pretty'))
    
def show_zeroshot():
    # models = ["BiomedCLIP", "CLIP", "PMCCLIP", "PubMedCLIP"] # zeroshot
    models = ["BiomedCLIP2", "CLIP2", "PMCCLIP2", "PubMedCLIP2"] # prompt+ensemble
    df = pd.DataFrame(index=datasets + ["avg"], columns=models)
    for dataset in datasets:
        for model in models:
            directory = f"output/{dataset}/Zeroshot{model}"
            acc, std, time = parse_function(metric, directory=directory) 
            df.loc[dataset, model] = acc     
    df.loc["avg"] = df.mean()
    print(tabulate(df, headers='keys', tablefmt='pretty'))
def show_base_model():
    models = ["BiomedCLIP", "CLIP", "PMCCLIP", "PubMedCLIP"]
    k1_data = {model: [] for model in models}
    k2_data = {model: [] for model in models}
    k4_data = {model: [] for model in models}
    k8_data = {model: [] for model in models}
    k16_data = {model: [] for model in models}
    avg_data = {model: [] for model in models}
    for dataset in datasets:
        for col in cols:
            data = {
                "k=1": [],
                "k=2": [],
                "k=4": [],
                "k=8": [],
                "k=16": [],
            }
            for model in models:
                all_time, all_avg = 0, 0
                for shot in shots:
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpend_lowminimal"
                    acc, std, time = parse_function(metric, directory=directory)       
                    # data[f'k={shot}'].append(f"{acc.round(2)}+- {std.round(2):.2f}%. ")  
                    data[f'k={shot}'].append(acc.round(2))  
            
            dfs = pd.DataFrame(data)
            dfs.index = models
            dfs.index.name = dataset
            print(tabulate(dfs, headers='keys', tablefmt='pretty'))
            for model in models:
                k1_data[model].append(dfs.loc[model, 'k=1'])
                k2_data[model].append(dfs.loc[model, 'k=2'])
                k4_data[model].append(dfs.loc[model, 'k=4'])
                k8_data[model].append(dfs.loc[model, 'k=8'])
                k16_data[model].append(dfs.loc[model, 'k=16'])
        # dfs.to_csv(f"output_train_csv/{file_names[dataset]}")
    k1_avg = {model: round(sum(k1_data[model]) / len(k1_data[model]), 2) for model in models}
    k2_avg = {model: round(sum(k2_data[model]) / len(k2_data[model]), 2) for model in models}
    k4_avg = {model: round(sum(k4_data[model]) / len(k4_data[model]), 2) for model in models}
    k8_avg = {model: round(sum(k8_data[model]) / len(k8_data[model]), 2) for model in models}
    k16_avg = {model: round(sum(k16_data[model]) / len(k16_data[model]), 2) for model in models}
     # 将 HM 平均值转换为 DataFrame 以便输出
    avg_df = pd.DataFrame({
        'Method': models,
        'k=1': [k1_avg[model] for model in models],
        'k=2': [k2_avg[model] for model in models],
        'k=4': [k4_avg[model] for model in models],
        'k=8': [k8_avg[model] for model in models],
        'k=16': [k16_avg[model] for model in models],
    })
    avg_df.set_index('Method', inplace=True)

    # 打印 HM 平均值表格
    print("\n每个方法的平均值：")
    print(tabulate(avg_df, headers='keys', tablefmt='pretty'))

'''
def show_few_shot_ctx():
    # ctx_list = ["end, CSC", "mid, CSC", "end", "mid"]
    ctx_list = ["mid, CTX", "front, CTX", "end, CTX", "end", "front", "mid", "end, CSC", "mid, CSC", "front, CSC"]
    col = "BiomedAP"
    model = "BiomedCLIP"
    k1_data = {ctx: [] for ctx in ctx_list}
    k2_data = {ctx: [] for ctx in ctx_list}
    k4_data = {ctx: [] for ctx in ctx_list}
    k8_data = {ctx: [] for ctx in ctx_list}
    k16_data = {ctx: [] for ctx in ctx_list}
    avg_data = {ctx: [] for ctx in ctx_list}
    
    for dataset in datasets:
        data = {
            "k=1": [],
            "k=2": [],
            "k=4": [],
            "k=8": [],
            "k=16": [],
        }
        for ctx in ctx_list:
            all_time, all_avg = 0, 0
            for shot in shots:
                if ctx == "end":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpend_lowminimal"
                elif ctx == "mid":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpmiddle_lowminimal"
                elif ctx == "front":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpfront_lowminimal"
                elif ctx == "end, CSC":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscTrue_ctpend_lowminimal"
                elif ctx == "mid, CSC":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscTrue_ctpmiddle_lowminimal"
                elif ctx == "front, CSC":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscTrue_ctpfront_lowminimal"
                elif ctx == "end, CTX":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpend_ctxinit"
                elif ctx == "mid, CTX":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpmiddle_ctxinit" 
                elif ctx == "front, CTX":
                    directory = f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpfront_ctxinit" 
                acc, std, time = parse_function(metric, directory=directory)       
                data[f'k={shot}'].append(acc.round(2))  
            
        dfs = pd.DataFrame(data, index=ctx_list)  # Set ctx_list as index
        
        # Calculate average across all shot values for each context
        dfs['avg'] = dfs.mean(axis=1).round(2)
        
        dfs.index.name = dataset
        print(tabulate(dfs, headers='keys', tablefmt='pretty'))
    
        for ctx in ctx_list:
            k1_data[ctx].append(dfs.loc[ctx, 'k=1'])
            k2_data[ctx].append(dfs.loc[ctx, 'k=2'])
            k4_data[ctx].append(dfs.loc[ctx, 'k=4'])
            k8_data[ctx].append(dfs.loc[ctx, 'k=8'])
            k16_data[ctx].append(dfs.loc[ctx, 'k=16'])
            avg_data[ctx].append(dfs.loc[ctx, 'avg'])
        # dfs.to_csv(f"output_train_csv/{file_names[dataset]}")
    k1_avg = {ctx: round(sum(k1_data[ctx]) / len(k1_data[ctx]), 2) for ctx in ctx_list}
    k2_avg = {ctx: round(sum(k2_data[ctx]) / len(k2_data[ctx]), 2) for ctx in ctx_list}
    k4_avg = {ctx: round(sum(k4_data[ctx]) / len(k4_data[ctx]), 2) for ctx in ctx_list}
    k8_avg = {ctx: round(sum(k8_data[ctx]) / len(k8_data[ctx]), 2) for ctx in ctx_list}
    k16_avg = {ctx: round(sum(k16_data[ctx]) / len(k16_data[ctx]), 2) for ctx in ctx_list}
    all_avg = {ctx: round(sum(avg_data[ctx]) / len(avg_data[ctx]), 2) for ctx in ctx_list}
     # 将 HM 平均值转换为 DataFrame 以便输出
    avg_df = pd.DataFrame({
        'Method': ctx_list,
        'k=1': [k1_avg[ctx] for ctx in ctx_list],
        'k=2': [k2_avg[ctx] for ctx in ctx_list],
        'k=4': [k4_avg[ctx] for ctx in ctx_list],
        'k=8': [k8_avg[ctx] for ctx in ctx_list],
        'k=16': [k16_avg[ctx] for ctx in ctx_list],
        'Average': [all_avg[ctx] for ctx in ctx_list],
    })
    avg_df.set_index('Method', inplace=True)
    print("\n每个方法的平均值：")
    print(tabulate(avg_df, headers='keys', tablefmt='pretty'))
'''

def show_few_shot_ctx():
    # ctx_list = ["end, CSC", "mid, CSC", "end", "mid"]
    # ctx_list = ["mid, CTX", "front, CTX", "end, CTX", "end", "front", "mid",
    #             "end, CSC", "mid, CSC", "front, CSC"]
    ctx_list = ["end"]
    col = "BiomedAP"
    model = "BiomedCLIP"
    lowtype = "minimal"
    expected_seeds = ["seed1", "seed2", "seed3"]

    out_dir = "output_train_csv/fewshot_ctx"
    os.makedirs(out_dir, exist_ok=True)

    def get_directory(dataset, shot, ctx):
        if ctx == "end":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpend_low{lowtype}"
        elif ctx == "mid":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpmiddle_low{lowtype}"
        elif ctx == "front":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpfront_low{lowtype}"
        elif ctx == "end, CSC":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscTrue_ctpend_low{lowtype}"
        elif ctx == "mid, CSC":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscTrue_ctpmiddle_low{lowtype}"
        elif ctx == "front, CSC":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscTrue_ctpfront_low{lowtype}"
        elif ctx == "end, CTX":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpend_ctxinit"
        elif ctx == "mid, CTX":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpmiddle_ctxinit"
        elif ctx == "front, CTX":
            return f"output/{dataset}/shots_{shot}/{col}_{model}/cscFalse_ctpfront_ctxinit"
        else:
            raise ValueError(f"Unknown ctx: {ctx}")

    all_detail_rows = []

    for dataset in datasets:
        rows = []

        for ctx in ctx_list:
            row = {"Context": ctx}

            shot_means = []
            shot_bests = []

            for shot in shots:  # shots = [1,2,4,8,16] :contentReference[oaicite:2]{index=2}
                directory = get_directory(dataset, shot, ctx)

                acc_mean, acc_ci95, _time, seedmap = parse_function(metric, directory=directory)

                # 记录每个 seed 的 accuracy
                seed_vals = []
                for sd in expected_seeds:
                    v = seedmap.get(sd, np.nan)
                    row[f"k={shot}_{sd}"] = (round(v, 2) if not np.isnan(v) else np.nan)
                    if not np.isnan(v):
                        seed_vals.append(v)

                # 记录 mean / ci95 / best
                row[f"k={shot}_mean"] = round(acc_mean, 2)
                row[f"k={shot}_ci95"] = round(acc_ci95, 2)
                row[f"k={shot}_best"] = (round(max(seed_vals), 2) if len(seed_vals) > 0 else np.nan)

                shot_means.append(acc_mean)
                if len(seed_vals) > 0:
                    shot_bests.append(max(seed_vals))

            # 额外：跨 k 的平均（可留可删）
            row["avg_mean"] = round(float(np.mean(shot_means)), 2) if len(shot_means) > 0 else np.nan
            row["avg_best"] = round(float(np.mean(shot_bests)), 2) if len(shot_bests) > 0 else np.nan

            rows.append(row)

        dfs = pd.DataFrame(rows).set_index("Context")
        dfs.index.name = dataset

        print(tabulate(dfs, headers="keys", tablefmt="pretty"))

        # 保存每个 dataset 的 CSV（每个 ctx 一行，包含所有 k/seed/mean/best）
        per_path = os.path.join(out_dir, f"{dataset}_fewshot_ctx_detail.csv")
        dfs.to_csv(per_path)
        print(f"已保存: {per_path}")

        # 汇总到总表（加 Dataset 列，便于后续透视）
        tmp = dfs.reset_index()
        tmp.insert(0, "Dataset", dataset)
        all_detail_rows.append(tmp)

    # 保存合并后的总 CSV
    if len(all_detail_rows) > 0:
        all_detail_df = pd.concat(all_detail_rows, ignore_index=True)
        all_path = os.path.join(out_dir, "fewshot_ctx_all_detail.csv")
        all_detail_df.to_csv(all_path, index=False)
        print(f"\n已保存合并明细CSV: {all_path}")

if __name__ == "__main__":    
    # show_base_to_new()
    # show_few_shot_coop()
    # show_zeroshot()
    # show_few_shot_adapter()
    # show_base_model()
    show_few_shot_ctx()