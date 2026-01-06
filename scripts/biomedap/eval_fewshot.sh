#!/bin/bash

# BiomedAP Few-shot Evaluation Script
# Usage: bash eval_fewshot.sh <DATA_PATH> <DATASET> <SHOTS>
# Example: bash eval_fewshot.sh data BTMRI 16

DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
LOADEP=100

# 【新增】BiomedAP 参数配置
LOW_TEMPLATE_TYPE=minimal
# L1_LAMBDA_HIGH=25.0
# L1_LAMBDA_LOW=0.3
# KL_LAMBDA=0.5

METHOD=BiomedAP
TRAINER=BIOMEDAP_${MODEL}

for SEED in 1 2 3
do
    MODEL_DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    if [ -d "$MODEL_DIR" ]; then
        echo "The checkpoint exists at ${MODEL_DIR} (skipping to evaluation)"
    else
        echo "Warning: Checkpoint not found at ${MODEL_DIR}"
        echo "Please train the model first using few_shot.sh"
        continue
    fi
    
    DIR=output_eval/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --output-dir ${DIR} \
            --eval-only \
            TRAINER.BIOMEDAP.N_CTX ${NCTX} \
            TRAINER.BIOMEDAP.CSC ${CSC} \
            TRAINER.BIOMEDAP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE}
            # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
            # TRAINER.BIOMEDAP.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
            # TRAINER.BIOMEDAP.KL_LAMBDA ${KL_LAMBDA}
    fi
done
