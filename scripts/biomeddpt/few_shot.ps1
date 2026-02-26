# custom config

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [int]$SHOTS,  # number of shots (1, 2, 4, 8, 16)
    
    [Parameter(Mandatory=$true)]
    [string]$MODEL
)

$NCTX = 4
$CSC = "False"
$CTP = "end"

$METHOD = "BiomedDPT"
$TRAINER = "BiomedDPT_$MODEL"

foreach ($SEED in 1, 2, 3) {
    $DIR = "output/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)"
    } else {
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/few_shot/$DATASET.yaml `
            --output-dir $DIR `
            TRAINER.BIOMEDCOOP.N_CTX $NCTX `
            TRAINER.BIOMEDCOOP.CSC $CSC `
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION $CTP `
            DATASET.NUM_SHOTS $SHOTS
    }
}
