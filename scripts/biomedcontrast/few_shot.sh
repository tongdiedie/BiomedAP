#!/bin/bash
# BiomedContrast Few-Shot Learning Script
# Usage: bash few_shot.sh DATA_PATH DATASET SHOTS MODEL

DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=$4
NCTX=4
CSC=False
CTP=end

METHOD=BiomedContrast
TRAINER=BiomedContrast_${MODEL}

for SEED in 1 2 3
do
    DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "======================================"
        echo "Few-Shot Training - Seed ${SEED}"
        echo "Dataset: ${DATASET}"
        echo "Shots: ${SHOTS}"
        echo "Method: ${METHOD} with negative repulsion"
        echo "======================================"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
        --output-dir ${DIR} \
        TRAINER.BIOMEDCONTRAST.N_CTX ${NCTX} \
        TRAINER.BIOMEDCONTRAST.CSC ${CSC} \
        TRAINER.BIOMEDCONTRAST.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

echo "======================================"
echo "Few-Shot training completed!"
echo "======================================"