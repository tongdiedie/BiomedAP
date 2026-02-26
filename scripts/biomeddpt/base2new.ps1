# custom config

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET,
    
    [Parameter(Mandatory=$true)]
    [string]$MODEL
)

$METHOD = "BiomedDPT"
$TRAINER = "BiomedDPT_$MODEL"

$SHOTS = 16
$LOADEP = 50
$CTP = "end"
$CSC = "False"
$NCTX = 4
$SUB_base = "base"
$SUB_novel = "new"

foreach ($SEED in 1, 2, 3) {
    $DIR = "output/base2new/train_$SUB_base/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)"
    } else {
        python train.py `
            --root $DATA `
            --seed $SEED `
            --trainer $TRAINER `
            --dataset-config-file configs/datasets/$DATASET.yaml `
            --config-file configs/trainers/$METHOD/base_to_novel/$DATASET.yaml `
            --output-dir $DIR `
            DATASET.NUM_SHOTS $SHOTS `
            DATASET.SUBSAMPLE_CLASSES $SUB_base
    }
    
    $COMMON_DIR = "$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    $MODEL_DIR = "output/base2new/train_$SUB_base/$COMMON_DIR"
    $DIR = "output/base2new/test_$SUB_novel/$COMMON_DIR"
    
    if (Test-Path $DIR) {
        Write-Host "Oops! The results exist at $DIR (so skip this job)"
    } else {
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
            DATASET.SUBSAMPLE_CLASSES $SUB_novel
    }
}
