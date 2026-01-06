<#
.SYNOPSIS
    BiomedDPT_Robust Base-to-New Evaluation Script
.DESCRIPTION
    Usage: .\eval_base2new.ps1 <DATA_PATH> <DATASET>
    Example: .\eval_base2new.ps1 data BTMRI
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET
)

# 配置参数
$SHOTS = 16
$MODEL = "BiomedCLIP"
$NCTX = 4
$CSC = "False"
$CTP = "end"
$LOADEP = 50

$METHOD = "BiomedDPT_Robust"
$TRAINER = "BiomedDPT_Robust_$MODEL"

# Robust 参数配置
$LOW_TEMPLATE_TYPE = "minimal"
# $L1_LAMBDA_HIGH = 0.5
# $L1_LAMBDA_LOW = 0.3
# $KL_LAMBDA = 0.1

# 评估 3 个不同的随机种子
foreach ($SEED in 1..3) {
    $MODEL_DIR = "base2new_robust\train_base\$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
    
    if (Test-Path $MODEL_DIR) {
        Write-Host "The checkpoint exists at $MODEL_DIR (skipping to evaluation)" -ForegroundColor Green
    }
    else {
        Write-Host "Warning: Checkpoint not found at $MODEL_DIR" -ForegroundColor Red
        Write-Host "Please train the model first using base2new.ps1" -ForegroundColor Yellow
        continue
    }
    
    # 评估 base 和 new 两个子任务
    foreach ($SUB_TASK in @("base", "new")) {
        $DIR = "output_eval\base2new\test_$SUB_TASK\$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
        
        if (Test-Path $DIR) {
            Write-Host "Oops! The results exist at $DIR (so skip this job)" -ForegroundColor Yellow
        }
        else {
            Write-Host "`nEvaluating on $SUB_TASK classes (SEED=$SEED)..." -ForegroundColor Cyan
            
            python train.py `
                --root $DATA `
                --seed $SEED `
                --trainer $TRAINER `
                --dataset-config-file configs/datasets/$DATASET.yaml `
                --config-file configs/trainers/$METHOD/base_to_novel/$DATASET.yaml `
                --model-dir $MODEL_DIR `
                --load-epoch $LOADEP `
                --output-dir $DIR `
                --eval-only `
                DATASET.SUBSAMPLE_CLASSES $SUB_TASK `
                TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
                # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH $L1_LAMBDA_HIGH `
                # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW $L1_LAMBDA_LOW `
                # TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA $KL_LAMBDA

        }
    }
}

Write-Host "`nAll seeds evaluation completed!" -ForegroundColor Green
