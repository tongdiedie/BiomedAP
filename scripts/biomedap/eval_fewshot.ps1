<#
.SYNOPSIS
    BiomedAP Few-shot Evaluation Script
.DESCRIPTION
    Usage: .\eval_fewshot.ps1 <DATA_PATH> <DATASET> <SHOTS>
    Example: .\eval_fewshot.ps1 data BTMRI 16
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [int]$SHOTS
)

# 配置参数
$MODEL = "BiomedCLIP"
$NCTX = 4
$CSC = "False"
$CTP = "end"
$LOADEP = 100

# Robust 参数配置
$LOW_TEMPLATE_TYPE = "minimal"
# $L1_LAMBDA_HIGH = 0.5
# $L1_LAMBDA_LOW = 0.3
# $KL_LAMBDA = 0.1

$METHOD = "BiomedAP"
$TRAINER = "BiomedAP_$MODEL"

# 评估 3 个不同的随机种子
foreach ($SEED in 1..3) {
    $MODEL_DIR = "few_shot_robust\$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
    
    if (Test-Path $MODEL_DIR) {
        Write-Host "The checkpoint exists at $MODEL_DIR (skipping to evaluation)" -ForegroundColor Green
    }
    else {
        Write-Host "Warning: Checkpoint not found at $MODEL_DIR" -ForegroundColor Red
        Write-Host "Please train the model first using few_shot.ps1" -ForegroundColor Yellow
        continue
    }
    
    $DIR = "output_eval\$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)" -ForegroundColor Yellow
    }
    else {
        Write-Host "`nEvaluating with SEED=$SEED..." -ForegroundColor Cyan
        
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/few_shot/$DATASET.yaml `
            --model-dir $MODEL_DIR `
            --load-epoch $LOADEP `
            --output-dir $DIR `
            --eval-only `
            TRAINER.BIOMEDAP.N_CTX $NCTX `
            TRAINER.BIOMEDAP.CSC $CSC `
            TRAINER.BIOMEDAP.CLASS_TOKEN_POSITION $CTP `
            TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
            # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH $L1_LAMBDA_HIGH `
            # TRAINER.BIOMEDAP.L1_LAMBDA_LOW $L1_LAMBDA_LOW `
            # TRAINER.BIOMEDAP.KL_LAMBDA $KL_LAMBDA

    }
}

Write-Host "`nAll seeds evaluation completed!" -ForegroundColor Green
