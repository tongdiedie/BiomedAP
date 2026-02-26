# BiomedContrast Few-Shot Learning Script (PowerShell)
# Usage: .\few_shot.ps1 -DATA "data" -DATASET "btmri" -SHOTS 16 -MODEL "BiomedCLIP"

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [int]$SHOTS,
    
    [Parameter(Mandatory=$true)]
    [string]$MODEL
)

$NCTX = 4
$CSC = "False"
$CTP = "end"

$METHOD = "BiomedContrast"
$TRAINER = "BiomedContrast_$MODEL"

foreach ($SEED in 1, 2, 3) {
    $DIR = "output/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)"
    } else {
        Write-Host "======================================"
        Write-Host "Few-Shot Training - Seed $SEED"
        Write-Host "Dataset: $DATASET"
        Write-Host "Shots: $SHOTS"
        Write-Host "Method: $METHOD with negative repulsion"
        Write-Host "======================================"
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/few_shot/$DATASET.yaml `
            --output-dir $DIR `
            TRAINER.BIOMEDCONTRAST.N_CTX $NCTX `
            TRAINER.BIOMEDCONTRAST.CSC $CSC `
            TRAINER.BIOMEDCONTRAST.CLASS_TOKEN_POSITION $CTP `
            DATASET.NUM_SHOTS $SHOTS
    }
}

Write-Host "======================================"
Write-Host "Few-Shot training completed!"
Write-Host "======================================"