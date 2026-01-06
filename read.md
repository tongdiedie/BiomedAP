🎯 核心改进
原始 BiomedDPT 损失函数
L = L_ce + λ1 * L_L1_high + λ2 * L_KL

改进的 BiomedDPT_Robust 损失函数
L = L_ce + λ1 * L_L1_high + λ2 * L_KL + λ3 * L_L1_low
                                         ↑
                                    【关键新增】


📁 项目结构
Biomed-prompt-learning/
├── trainers/
│   ├── prompt_templates.py              # Prompt 模板定义（高质量、低质量）
│   ├── BiomedDPT/                       # 原始 BiomedDPT（不修改）
│   └── BiomedDPT_Robust/                # 【新增】鲁棒性增强版
│       ├── biomeddpt_robust_clip.py
│       ├── biomeddpt_robust_biomedclip.py
│       ├── biomeddpt_robust_pubmedclip.py
│       └── biomeddpt_robust_pmcclip.py
│
├── configs/
│   ├── datasets/                        # 数据集配置
│   │   ├── btmri.yaml
│   │   ├── busi.yaml
│   │   └── ...
│   └── trainers/
│       ├── BiomedDPT/                   # 原始 BiomedDPT
│       └── BiomedDPT_Robust/            # 【新增】
│           ├── few_shot/                # Few-Shot 实验配置（11 个文件）
│           └── base_to_novel/           # Base-to-Novel 实验配置（11 个文件）
│
├── scripts/
│   ├── few_shot_robust.sh               # Linux/Mac 运行脚本
│   ├── few_shot_robust.ps1              # Windows PowerShell 运行脚本
│   └── quick_test_robust.ps1            # 快速测试脚本
│
├── train.py                              # 训练主入口
└── README.md                             # 本文档

🚀 快速开始
1. 准备数据集

将数据集放在 data/ 目录下，例如：

data/
├── BTMRI/
│   ├── train/
│   ├── val/
│   └── test/
├── BUSI/
└── ...

2. 快速测试（5 epochs）
# Windows PowerShell
.\scripts\quick_test_robust.ps1

# Linux/Mac
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root data \
    --trainer BiomedDPT_Robust_BiomedCLIP \
    --dataset-config-file configs/datasets/btmri.yaml \
    --config-file configs/trainers/BiomedDPT_Robust/few_shot/btmri.yaml \
    --output-dir output/quick_test \
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 \
    DATASET.NUM_SHOTS 16 \
    OPTIM.MAX_EPOCH 5

3. 完整 Few-Shot 实验
# Windows PowerShell
.\scripts\few_shot_robust.ps1 -DATA "data" -DATASET "btmri" -SHOTS 16 -BACKBONE "BiomedCLIP"

# Linux/Mac
bash scripts/few_shot_robust.sh data btmri 16 BiomedCLIP

⚙️ 参数配置
1. 核心参数（必须调整）
损失权重参数
参数	默认值	范围	说明
L1_LAMBDA_HIGH	数据集特定	5.0-70.0	向高质量 Prompt 对齐的权重（λ1）
KL_LAMBDA	数据集特定	0.1-20.0	知识蒸馏权重（λ2）
L1_LAMBDA_LOW	0.3	0.0-0.7	向低质量 Prompt 对齐的权重（λ3）【关键新增】

调节建议：

λ3 = 0.0：退化为原始 BiomedDPT（baseline）

λ3 = 0.1-0.2：弱约束，轻微引导

λ3 = 0.3-0.4：中等约束，推荐值

λ3 = 0.5-0.7：强约束，可能过度简化

低质量 Prompt 模板类型
参数值	示例	说明
minimal	"glioma"	仅类别名（推荐）
article	"a glioma"	加冠词
generic	"a photo of glioma"	通用描述
medical_minimal	"a medical image of glioma"	极简医学术语
empty	""	空字符串（极端情况）
2. Prompt 设置参数
参数	默认值	说明
N_CTX	4-6	Prompt 长度（上下文向量数量）
CTX_INIT	数据集特定	初始化文本（如 "a MR photo of a"）
N_PROMPTS	50	使用的 GPT-4 高质量 Prompt 数量
PREC	"fp32"	精度（"fp16", "fp32", "amp"）
3. 训练超参数
参数	Few-Shot	Base-to-Novel	说明
MAX_EPOCH	50	100	训练轮次
LR	0.0025	0.0025	学习率
BATCH_SIZE	4	4	批大小
NUM_SHOTS	16	-	Few-Shot 每类样本数
4. 配置文件位置
# 配置文件路径：configs/trainers/BiomedDPT_Robust/few_shot/btmri.yaml

TRAINER:
  BIOMEDDPT_ROBUST:
    # Prompt 设置
    CTX_INIT: "a MR photo of a"
    N_CTX: 5
    PREC: "fp32"
    N_PROMPTS: 50
    
    # 损失权重（可调参数）
    L1_LAMBDA_HIGH: 12.5      # λ1：向高质量对齐
    KL_LAMBDA: 0.25           # λ2：知识蒸馏
    L1_LAMBDA_LOW: 0.3        # λ3：向低质量对齐【关键新增】
    
    # 低质量 Prompt 模板
    LOW_TEMPLATE_TYPE: "minimal"  # 可选值见上表

🧪 实验类型
1. Few-Shot 学习

目标：在少量标注样本（如 16-shot）下训练模型

配置文件：configs/trainers/BiomedDPT_Robust/few_shot/*.yaml

运行命令：

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root data \
    --trainer BiomedDPT_Robust_BiomedCLIP \
    --dataset-config-file configs/datasets/btmri.yaml \
    --config-file configs/trainers/BiomedDPT_Robust/few_shot/btmri.yaml \
    --output-dir output/btmri_16shot \
    DATASET.NUM_SHOTS 16

2. Base-to-Novel 泛化

目标：在 base 类上训练，测试在 novel 类（未见过的类别）上的泛化能力

配置文件：configs/trainers/BiomedDPT_Robust/base_to_novel/*.yaml

训练（仅 base 类）：

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root data \
    --trainer BiomedDPT_Robust_BiomedCLIP \
    --dataset-config-file configs/datasets/btmri.yaml \
    --config-file configs/trainers/BiomedDPT_Robust/base_to_novel/btmri.yaml \
    --output-dir output/btmri_base_to_novel \
    DATASET.SUBSAMPLE_CLASSES base


测试（base 类）：

python train.py \
    --root data \
    --trainer BiomedDPT_Robust_BiomedCLIP \
    --dataset-config-file configs/datasets/btmri.yaml \
    --config-file configs/trainers/BiomedDPT_Robust/base_to_novel/btmri.yaml \
    --model-dir output/btmri_base_to_novel \
    --load-epoch 100 \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES base


测试（novel 类）：

python train.py \
    --root data \
    --trainer BiomedDPT_Robust_BiomedCLIP \
    --dataset-config-file configs/datasets/btmri.yaml \
    --config-file configs/trainers/BiomedDPT_Robust/base_to_novel/btmri.yaml \
    --model-dir output/btmri_base_to_novel \
    --load-epoch 100 \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES new

📜 运行脚本
1. PowerShell 脚本（Windows）
Few-Shot 实验脚本
# 文件：scripts/few_shot_robust.ps1

# 运行方式
.\scripts\few_shot_robust.ps1 `
    -DATA "data" `
    -DATASET "btmri" `
    -SHOTS 16 `
    -BACKBONE "BiomedCLIP"

# 参数说明：
# -DATA: 数据根目录
# -DATASET: 数据集名称（btmri, busi, covid, 等）
# -SHOTS: Few-shot 样本数（1, 2, 4, 8, 16）
# -BACKBONE: 模型骨干网络（BiomedCLIP, CLIP, PubMedCLIP, PMCCLIP）

快速测试脚本
# 文件：scripts/quick_test_robust.ps1

# 运行方式（5 epochs 快速验证）
.\scripts\quick_test_robust.ps1

2. Bash 脚本（Linux/Mac）
# 文件：scripts/few_shot_robust.sh

# 运行方式
chmod +x scripts/few_shot_robust.sh
bash scripts/few_shot_robust.sh data btmri 16 BiomedCLIP

# 参数说明：
# $1: 数据根目录
# $2: 数据集名称
# $3: Few-shot 样本数
# $4: 模型骨干网络

📊 结果分析
1. 查看训练日志

训练日志保存在 output/ 目录下：

output/
├── btmri_16shot/
│   ├── log.txt                  # 训练日志
│   ├── prompt_learner/
│   │   └── model-best.pth.tar   # 最佳模型
│   └── ...


日志内容示例：

Epoch 1/100
loss: 2.345, loss_ce: 1.234, loss_l1_high: 0.456, loss_kl: 0.321, loss_l1_low: 0.234
acc: 52.3%

...

Epoch 100/100
loss: 0.543, loss_ce: 0.321, loss_l1_high: 0.098, loss_kl: 0.067, loss_l1_low: 0.057
acc: 87.5%

2. 对比实验结果
实验	描述	预期准确率	输出目录
Baseline	原始 BiomedDPT（λ3=0）	85.0%	output/btmri_baseline
Robust (minimal)	λ3=0.3，低质量="glioma"	87.0% ↑	output/btmri_robust_minimal
Robust (article)	λ3=0.3，低质量="a glioma"	86.5% ↑	output/btmri_robust_article
Robust (empty)	λ3=0.3，低质量=""	86.0% ↑	output/btmri_robust_empty
3. 可视化分析

使用 TensorBoard 查看训练曲线：

tensorboard --logdir output/

🔬 消融实验
1. λ3（L1_LAMBDA_LOW）权重消融

测试不同的 λ3 权重对性能的影响：

# λ3 = 0.0 (Baseline)
python train.py ... TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.0

# λ3 = 0.1
python train.py ... TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.1

# λ3 = 0.3 (推荐)
python train.py ... TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3

# λ3 = 0.5
python train.py ... TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.5

# λ3 = 0.7
python train.py ... TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.7


预期结果：

λ3	准确率	说明
0.0	85.0%	Baseline
0.1	85.8%	轻微改善
0.3	87.0%	最佳
0.5	86.5%	过强
0.7	85.5%	过度简化
2. 低质量 Prompt 模板类型消融
# minimal（推荐）
python train.py ... TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal

# article
python train.py ... TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE article

# generic
python train.py ... TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE generic

# empty
python train.py ... TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE empty

3. 不同 Backbone 对比
# BiomedCLIP（推荐）
python train.py --trainer BiomedDPT_Robust_BiomedCLIP ...

# CLIP
python train.py --trainer BiomedDPT_Robust_CLIP ...

# PubMedCLIP
python train.py --trainer BiomedDPT_Robust_PubMedCLIP ...

# PMC-CLIP
python train.py --trainer BiomedDPT_Robust_PMCCLIP ...

❓ 常见问题
Q1: 如何选择 λ3（L1_LAMBDA_LOW）的值？

A:

从 0.3 开始（推荐默认值）

如果效果不明显，尝试 0.4-0.5（加强约束）

如果性能下降，尝试 0.1-0.2（减弱约束）

避免 > 0.7（过强会损害性能）

Q2: 如何选择低质量 Prompt 模板类型？

A:

推荐使用 minimal（仅类别名）：效果最好，约束最强

article 和 generic：适度约束，适合某些数据集

empty：仅用于极端消融实验

Q3: 训练时显存不足怎么办？

A:

# 方法 1：减小批大小
DATALOADER.TRAIN_X.BATCH_SIZE 2

# 方法 2：使用 fp16 精度
TRAINER.BIOMEDDPT_ROBUST.PREC fp16

# 方法 3：减少 GPT-4 Prompt 数量
TRAINER.BIOMEDDPT_ROBUST.N_PROMPTS 25

Q4: 如何添加新的数据集？

A:

准备数据集，放在 data/YOUR_DATASET/

创建数据集配置 configs/datasets/your_dataset.yaml

创建训练配置 configs/trainers/BiomedDPT_Robust/few_shot/your_dataset.yaml

在 trainers/prompt_templates.py 中添加数据集的 GPT-4 Prompt 和模板

Q5: 如何使用预训练模型？

A:

python train.py \
    --model-dir output/btmri_16shot \
    --load-epoch 100 \
    --eval-only \
    ...

Q6: 如何对比 BiomedDPT 和 BiomedDPT_Robust？

A:

# 1. 运行原始 BiomedDPT
python train.py --trainer BiomedDPT_BiomedCLIP ...

# 2. 运行 BiomedDPT_Robust（λ3=0.0，相当于 baseline）
python train.py --trainer BiomedDPT_Robust_BiomedCLIP ... \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.0

# 3. 运行 BiomedDPT_Robust（λ3=0.3，启用低质量约束）
python train.py --trainer BiomedDPT_Robust_BiomedCLIP ... \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3

# 4. 对比三个实验的 log.txt 文件

📚 参考文献