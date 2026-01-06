#!/bin/bash

# Quick Test Script for BiomedDPT_Robust
# Usage: bash quick_test_robust.sh

DATA=data
DATASET=BTMRI
SHOTS=16
MODEL=BiomedCLIP
SEED=1

LOW_TEMPLATE_TYPE=minimal
L1_LAMBDA_LOW=0.3

METHOD=BiomedDPT_Robust
TRAINER=BiomedDPT_Robust_${MODEL}

DIR=output/quick_test_robust

echo "Running quick test (5 epochs)..."

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
    --output-dir ${DIR} \
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH 5

echo "Quick test completed!"


# <#
# .SYNOPSIS
#     Quick Test Script for BiomedDPT_Robust (Advanced)
# .DESCRIPTION
#     快速测试脚本，支持自定义参数
#     Usage: 
#         .\quick_test.ps1
#         .\quick_test.ps1 -DATASET BUSI -MODEL CLIP -EPOCHS 10
# .PARAMETER DATASET
#     数据集名称，默认: BTMRI
# .PARAMETER MODEL
#     Backbone 模型，默认: BiomedCLIP
# .PARAMETER EPOCHS
#     训练轮数，默认: 5
# .PARAMETER SHOTS
#     Few-shot 样本数，默认: 16
# #>

# param(
#     [string]$DATASET = "BTMRI",
    
#     [ValidateSet("BiomedCLIP", "CLIP", "PubMedCLIP", "PMCCLIP")]
#     [string]$MODEL = "BiomedCLIP",
    
#     [int]$EPOCHS = 5,
    
#     [ValidateSet(1, 2, 4, 8, 16)]
#     [int]$SHOTS = 16,
    
#     [int]$SEED = 1,
    
#     [ValidateSet("minimal", "class_only", "empty")]
#     [string]$LOW_TEMPLATE_TYPE = "minimal",
    
#     [double]$L1_LAMBDA_LOW = 0.3
# )

# # 固定配置
# $DATA = "data"
# $METHOD = "BiomedDPT_Robust"
# $TRAINER = "BiomedDPT_Robust_$MODEL"
# $DIR = "output\quick_test_robust_${DATASET}_${MODEL}_${EPOCHS}ep"

# Write-Host "`n========================================" -ForegroundColor Cyan
# Write-Host "   BiomedDPT_Robust Quick Test" -ForegroundColor Cyan
# Write-Host "========================================" -ForegroundColor Cyan
# Write-Host "Dataset:           $DATASET" -ForegroundColor Yellow
# Write-Host "Backbone:          $MODEL" -ForegroundColor Yellow
# Write-Host "Shots:             $SHOTS" -ForegroundColor Yellow
# Write-Host "Epochs:            $EPOCHS" -ForegroundColor Yellow
# Write-Host "Seed:              $SEED" -ForegroundColor Yellow
# Write-Host "Low Template:      $LOW_TEMPLATE_TYPE" -ForegroundColor Yellow
# Write-Host "L1 Lambda Low:     $L1_LAMBDA_LOW" -ForegroundColor Yellow
# Write-Host "Output Dir:        $DIR" -ForegroundColor Yellow
# Write-Host "========================================`n" -ForegroundColor Cyan

# # 检查数据集配置文件是否存在
# $datasetConfig = "configs\datasets\$DATASET.yaml"
# if (-not (Test-Path $datasetConfig)) {
#     Write-Host "Error: Dataset config file not found: $datasetConfig" -ForegroundColor Red
#     exit 1
# }

# # 检查训练器配置文件是否存在
# $trainerConfig = "configs\trainers\$METHOD\few_shot\$DATASET.yaml"
# if (-not (Test-Path $trainerConfig)) {
#     Write-Host "Error: Trainer config file not found: $trainerConfig" -ForegroundColor Red
#     exit 1
# }

# Write-Host "Starting training..." -ForegroundColor Green

# python train.py `
#     --root $DATA `
#     --seed $SEED `
#     --trainer $TRAINER `
#     --dataset-config-file $datasetConfig `
#     --config-file $trainerConfig `
#     --output-dir $DIR `
#     TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
#     TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW $L1_LAMBDA_LOW `
#     DATASET.NUM_SHOTS $SHOTS `
#     OPTIM.MAX_EPOCH $EPOCHS

# if ($LASTEXITCODE -eq 0) {
#     Write-Host "`n========================================" -ForegroundColor Green
#     Write-Host "   Quick test completed successfully!" -ForegroundColor Green
#     Write-Host "========================================" -ForegroundColor Green
#     Write-Host "Results saved to: $DIR`n" -ForegroundColor Green
# } else {
#     Write-Host "`n========================================" -ForegroundColor Red
#     Write-Host "   Quick test failed!" -ForegroundColor Red
#     Write-Host "========================================" -ForegroundColor Red
#     Write-Host "Please check the error messages above.`n" -ForegroundColor Red
# }
