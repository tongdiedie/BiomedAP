# ========================================
# BiomedTriplePrompt 训练脚本
# 文件位置：scripts/run_biomedtripleprompt.ps1
#
# 功能：运行三层级 Prompt 学习实验
# ========================================

# 设置通用参数
$DATA_ROOT = "data"
$DATASET = "btmri"
$SHOTS = 16
$TRAINER = "BiomedTriplePrompt"
$CONFIG = "configs/trainers/BiomedTriplePrompt/btmri.yaml"
$DATASET_CONFIG = "configs/datasets/btmri.yaml"

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "  BiomedTriplePrompt 三层级 Prompt 学习" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "`n数据集: $DATASET" -ForegroundColor Yellow
Write-Host "Few-shot: $SHOTS shots" -ForegroundColor Yellow
Write-Host "`n"

# ========== 实验 1：低质量模板 = minimal（仅类别名）==========
Write-Host "【实验 1】低质量模板 = minimal（仅类别名）" -ForegroundColor Green
Write-Host "输出目录: output/btmri_triple_minimal" -ForegroundColor Yellow

$env:CUDA_VISIBLE_DEVICES = "0"
python train.py `
    --root $DATA_ROOT `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CONFIG `
    --config-file $CONFIG `
    --output-dir output/btmri_triple_minimal `
    TRAINER.BIOMEDTRIPLEPROMPT.LOW_TEMPLATE_TYPE minimal `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 1 完成`n" -ForegroundColor Green


# ========== 实验 2：低质量模板 = article（a + 类别名）==========
Write-Host "【实验 2】低质量模板 = article（a + 类别名）" -ForegroundColor Green
Write-Host "输出目录: output/btmri_triple_article" -ForegroundColor Yellow

python train.py `
    --root $DATA_ROOT `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CONFIG `
    --config-file $CONFIG `
    --output-dir output/btmri_triple_article `
    TRAINER.BIOMEDTRIPLEPROMPT.LOW_TEMPLATE_TYPE article `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 2 完成`n" -ForegroundColor Green


# ========== 实验 3：低质量模板 = generic（a photo of）==========
Write-Host "【实验 3】低质量模板 = generic（a photo of）" -ForegroundColor Green
Write-Host "输出目录: output/btmri_triple_generic" -ForegroundColor Yellow

python train.py `
    --root $DATA_ROOT `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CONFIG `
    --config-file $CONFIG `
    --output-dir output/btmri_triple_generic `
    TRAINER.BIOMEDTRIPLEPROMPT.LOW_TEMPLATE_TYPE generic `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 3 完成`n" -ForegroundColor Green


# ========== 实验 4：低质量模板 = empty（空字符串）==========
Write-Host "【实验 4】低质量模板 = empty（空字符串，极端情况）" -ForegroundColor Green
Write-Host "输出目录: output/btmri_triple_empty" -ForegroundColor Yellow

python train.py `
    --root $DATA_ROOT `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CONFIG `
    --config-file $CONFIG `
    --output-dir output/btmri_triple_empty `
    TRAINER.BIOMEDTRIPLEPROMPT.LOW_TEMPLATE_TYPE empty `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 4 完成`n" -ForegroundColor Green


Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "  所有实验完成！" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "`n结果对比（查看各实验目录下的 log.txt）：" -ForegroundColor Yellow
Write-Host "1. Minimal: output/btmri_triple_minimal" -ForegroundColor White
Write-Host "2. Article: output/btmri_triple_article" -ForegroundColor White
Write-Host "3. Generic: output/btmri_triple_generic" -ForegroundColor White
Write-Host "4. Empty: output/btmri_triple_empty`n" -ForegroundColor White
