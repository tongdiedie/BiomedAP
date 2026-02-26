<#
.SYNOPSIS
    BiomedAP Few-shot Training Script
.DESCRIPTION
    Usage: .\few_shot.ps1 <DATA_PATH> <DATASET> <SHOTS> <MODEL>
    Example: .\few_shot.ps1 data btmri 16 BiomedCLIP
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [int]$SHOTS,
    
    [Parameter(Mandatory=$true)]
    [ValidateSet("BiomedCLIP", "CLIP", "PubMedCLIP", "PMCCLIP")]
    [string]$MODEL
)

# 配置参数
$NCTX = 5
$CSC = "False"
$CTP = "end"

# 【新增】Robust 参数配置
$LOW_TEMPLATE_TYPE = "minimal"  # 低质量模板类型: minimal, article, empty, medical_minimal, generic


$METHOD = "BiomedAP"
$TRAINER = "BiomedAP_$MODEL"

# 训练 3 个不同的随机种子
foreach ($SEED in 1..3) {
    $DIR = "output\$DATASET\shots_$SHOTS\$TRAINER\nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}\seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)" -ForegroundColor Yellow
    }
    else {
        Write-Host "`nTraining with SEED=$SEED..." -ForegroundColor Cyan
        
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/few_shot/$DATASET.yaml `
            --output-dir $DIR `
            TRAINER.BIOMEDAP.N_CTX $NCTX `
            TRAINER.BIOMEDAP.CSC $CSC `
            TRAINER.BIOMEDAP.CLASS_TOKEN_POSITION $CTP `
            TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE $LOW_TEMPLATE_TYPE `
            DATASET.NUM_SHOTS $SHOTS
    }
}

Write-Host "`nAll seeds training completed!" -ForegroundColor Green
