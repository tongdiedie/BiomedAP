#!/bin/bash
# BiomedContrast Base-to-New Evaluation Script
# Usage: bash eval_base2new.sh DATA_PATH DATASET

DATA=$1
DATASET=$2
SHOTS=16
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
LOADEP=50

METHOD=BiomedContrast
TRAINER=BiomedContrast_${MODEL}
SUB_base=base
SUB_novel=new

for SEED in 1 2 3
do
    MODEL_DIR=base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$MODEL_DIR" ]; then
        echo "The checkpoint exists at ${MODEL_DIR} (skipping to evaluation)"
    else
        echo "Checkpoint not found. Please train the model first or download checkpoints."
        python download_ckpts.py \
        --task base2new \
        --dataset ${DATASET} \
        --trainer ${TRAINER}
        echo "Downloaded the checkpoint for ${MODEL_DIR}"
    fi
    
    # 在base和new类上分别评估
    for SUB_TASK in "base" "new"
    do
        DIR=output_eval/base2new/test_${SUB_TASK}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            echo "======================================"
            echo "Evaluating on ${SUB_TASK} classes - Seed ${SEED}"
            echo "======================================"
            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --output-dir ${DIR} \
                --eval-only \
                DATASET.SUBSAMPLE_CLASSES ${SUB_TASK}
        fi
    done
done