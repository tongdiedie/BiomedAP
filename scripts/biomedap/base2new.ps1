<#
.SYNOPSIS
    BiomedAP Base-to-New Training and Evaluation Script
.DESCRIPTION
    Usage: .\base2new.ps1 <DATA_PATH> <DATASET> <MODEL>
    Example: .\base2new.ps1 data btmri BiomedCLIP
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [ValidateSet("BiomedCLIP", "CLIP", "PubMedCLIP", "PMCCLIP")]
    [string]$MODEL
)

# 配置参数
$SHOTS = 16
$LOADEP = 50
$CTP = "end"
$CSC = "False"
$NCTX = 5
$SUB_base = "base"
$SUB_novel = "new"

# Robust 参数配置
$LOW_TEMPLATE_TYPE = "minimal" # minimal/article/generic/medical_minimal/empty
# $L1_LAMBDA_HIGH = 0.5
# $L1_LAMBDA_LOW = 0.3
# $KL_LAMBDA = 0.1

$METHOD = "BiomedAP"
$TRAINER = "BiomedAP_$MODEL"

# 训练+评估 3 个不同的随机种子
foreach ($SEED in 1..3) {
    # ========== 阶段1：在 base classes 上训练 ==========
    $DIR = "output\base2new\train_$SUB_base\$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)" -ForegroundColor Yellow
    }
    else {
        Write-Host "`n========== Phase 1: Training on base classes (SEED=$SEED) ==========" -ForegroundColor Cyan
        
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/base_to_novel/$DATASET.yaml `
            --output-dir $DIR `
            DATASET.NUM_SHOTS $SHOTS `
            DATASET.SUBSAMPLE_CLASSES $SUB_base `
            TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
            # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH $L1_LAMBDA_HIGH `
            # TRAINER.BIOMEDAP.L1_LAMBDA_LOW $L1_LAMBDA_LOW `
            # TRAINER.BIOMEDAP.KL_LAMBDA $KL_LAMBDA

    }
    
    # ========== 阶段2：在 new classes 上评估 ==========
    $COMMON_DIR = "$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
    $MODEL_DIR = "output\base2new\train_$SUB_base\$COMMON_DIR"
    $DIR = "output\base2new\test_$SUB_novel\$COMMON_DIR"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)" -ForegroundColor Yellow
    }
    else {
        Write-Host "`n========== Phase 2: Evaluating on new classes (SEED=$SEED) ==========" -ForegroundColor Cyan
        
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/base_to_novel/$DATASET.yaml `
            --output-dir $DIR `
            --model-dir $MODEL_DIR `
            --load-epoch $LOADEP `
            --eval-only `
            DATASET.NUM_SHOTS $SHOTS `
            DATASET.SUBSAMPLE_CLASSES $SUB_novel `
            TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
            # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH $L1_LAMBDA_HIGH `
            # TRAINER.BIOMEDAP.L1_LAMBDA_LOW $L1_LAMBDA_LOW `
            # TRAINER.BIOMEDAP.KL_LAMBDA $KL_LAMBDA

    }
}

Write-Host "`nAll seeds training and evaluation completed!" -ForegroundColor Green
