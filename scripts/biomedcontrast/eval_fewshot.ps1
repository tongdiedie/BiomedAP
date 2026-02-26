# BiomedContrast Few-Shot Evaluation Script (PowerShell)

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [int]$SHOTS
)

$MODEL = "BiomedCLIP"
$NCTX = 4
$CSC = "False"
$CTP = "end"
$LOADEP = 100

$METHOD = "BiomedContrast"
$TRAINER = "BiomedContrast_$MODEL"

foreach ($SEED in 1, 2, 3) {
    $MODEL_DIR = "few_shot/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    
    if (Test-Path $MODEL_DIR) {
        Write-Host "The checkpoint exists at $MODEL_DIR (skipping to evaluation)"
    } else {
        python download_ckpts.py `
            --task few_shot `
            --dataset $DATASET `
            --shots $SHOTS `
            --trainer $TRAINER
        Write-Host "Downloaded the checkpoint for $MODEL_DIR"
    }
    
    $DIR = "output_eval/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)"
    } else {
        Write-Host "======================================"
        Write-Host "Few-Shot Evaluation - Seed $SEED"
        Write-Host "Dataset: $DATASET, Shots: $SHOTS"
        Write-Host "======================================"
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
            TRAINER.BIOMEDCONTRAST.N_CTX $NCTX `
            TRAINER.BIOMEDCONTRAST.CSC $CSC `
            TRAINER.BIOMEDCONTRAST.CLASS_TOKEN_POSITION $CTP
    }
}