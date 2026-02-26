<#
.SYNOPSIS
    Quick Test Script for BiomedAP (Base-to-New)
.DESCRIPTION
    快速测试 Base-to-New 任务（训练5个epoch验证代码）
    Usage: .\quick_test_base2new.ps1
#>

# 配置参数
$DATA = "data"
$DATASET = "BTMRI"
$MODEL = "BiomedCLIP"
$SEED = 1
$SHOTS = 16
$TRAIN_EPOCHS = 50 # 快速测试只训练5个epoch（可改）
$LOADEP = 50

# Robust 参数
$LOW_TEMPLATE_TYPE = "minimal"
$L1_LAMBDA_LOW = 0.3

$METHOD = "BiomedAP"
$TRAINER = "BiomedAP_$MODEL"

# ========== 阶段1：在 base classes 上训练 ==========
$TRAIN_DIR = "output\quick_test_base2new\train_base"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Phase 1: Training on base classes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dataset:       $DATASET" -ForegroundColor Yellow
Write-Host "Model:         $MODEL" -ForegroundColor Yellow
Write-Host "Shots:         $SHOTS" -ForegroundColor Yellow
Write-Host "Epochs:        $TRAIN_EPOCHS" -ForegroundColor Yellow
Write-Host "Seed:          $SEED" -ForegroundColor Yellow
Write-Host "Low Template:  $LOW_TEMPLATE_TYPE" -ForegroundColor Yellow
Write-Host "Output:        $TRAIN_DIR" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

python train.py `
    --root $DATA `
    --seed $SEED `
    --trainer $TRAINER `
    --dataset-config-file configs/datasets/$DATASET.yaml `
    --config-file configs/trainers/$METHOD/base_to_novel/$DATASET.yaml `
    --output-dir $TRAIN_DIR `
    DATASET.NUM_SHOTS $SHOTS `
    DATASET.SUBSAMPLE_CLASSES base `
    TRAINER.BiomedAP.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
    TRAINER.BiomedAP.L1_LAMBDA_LOW $L1_LAMBDA_LOW `
    OPTIM.MAX_EPOCH $TRAIN_EPOCHS

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPhase 1 training failed!" -ForegroundColor Red
    exit 1
}

# ========== 阶段2：在 new classes 上评估 ==========
$EVAL_DIR = "output\quick_test_base2new\test_new"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Phase 2: Evaluating on new classes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model Dir:     $TRAIN_DIR" -ForegroundColor Yellow
Write-Host "Load Epoch:    $LOADEP" -ForegroundColor Yellow
Write-Host "Output:        $EVAL_DIR" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

python train.py `
    --root $DATA `
    --seed $SEED `
    --trainer $TRAINER `
    --dataset-config-file configs/datasets/$DATASET.yaml `
    --config-file configs/trainers/$METHOD/base_to_novel/$DATASET.yaml `
    --model-dir $TRAIN_DIR `
    --load-epoch $LOADEP `
    --output-dir $EVAL_DIR `
    --eval-only `
    DATASET.NUM_SHOTS $SHOTS `
    DATASET.SUBSAMPLE_CLASSES new `
    TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
    TRAINER.BIOMEDAP.L1_LAMBDA_LOW $L1_LAMBDA_LOW

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  Quick test completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Training results:   $TRAIN_DIR" -ForegroundColor Green
    Write-Host "Evaluation results: $EVAL_DIR`n" -ForegroundColor Green
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "  Phase 2 evaluation failed!" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Red
}
