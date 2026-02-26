# custom config

param(
    [Parameter(Mandatory=$true)]
    [string]$DATA,
    
    [Parameter(Mandatory=$true)]
    [string]$DATASET
)

$SHOTS = 16
$MODEL = "BiomedCLIP"
$NCTX = 4
$CSC = "False"
$CTP = "end"
$LOADEP = 50

$METHOD = "BiomedDPT"
$TRAINER = "BiomedDPT_$MODEL"
$SUB_base = "base"
$SUB_novel = "new"

foreach ($SEED in 1, 2, 3) {
    $MODEL_DIR = "base2new/train_base/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
    
    if (Test-Path $MODEL_DIR) {
        Write-Host "The checkpoint exists at $MODEL_DIR (skipping to evaluation)"
    } else {
        python download_ckpts.py `
            --task base2new `
            --dataset $DATASET `
            --trainer $TRAINER
        Write-Host "Downloaded the checkpoint for $MODEL_DIR"
    }
    
    foreach ($SUB_TASK in "base", "new") {
        $DIR = "output_eval/base2new/test_$SUB_TASK/$DATASET/shots_$SHOTS/$TRAINER/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed$SEED"
        
        if (Test-Path $DIR) {
            Write-Host "Oops! The results exist at $DIR (so skip this job)"
        } else {
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
                DATASET.SUBSAMPLE_CLASSES $SUB_TASK
        }
    }
}
