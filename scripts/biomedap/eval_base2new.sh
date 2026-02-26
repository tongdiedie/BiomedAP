#!/bin/bash

# BiomedAP Base-to-New Evaluation Script
# Usage: bash eval_base2new.sh <DATA_PATH> <DATASET>
# Example: bash eval_base2new.sh data BTMRI

DATA=$1
DATASET=$2
SHOTS=16
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
LOADEP=50

METHOD=BiomedAP
TRAINER=BiomedAP_${MODEL}
SUB_base=base
SUB_novel=new

# 【新增】BiomedAP 参数配置
LOW_TEMPLATE_TYPE=minimal
# L1_LAMBDA_HIGH=25.0
# L1_LAMBDA_LOW=0.3
# KL_LAMBDA=0.5

export CUDA_VISIBLE_DEVICES=0

for SEED in 1 2 3
do
    MODEL_DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    if [ -d "$MODEL_DIR" ]; then
        echo "The checkpoint exists at ${MODEL_DIR} (skipping to evaluation)"
    else
        echo "Warning: Checkpoint not found at ${MODEL_DIR}"
        echo "Please train the model first using base2new.sh"
        continue
    fi
    
    for SUB_TASK in "base" "new"
    do
        DIR=output_eval/base2new/test_${SUB_TASK}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
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
                DATASET.SUBSAMPLE_CLASSES ${SUB_TASK} \
                TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE}
                # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
                # TRAINER.BIOMEDAP.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
                # TRAINER.BIOMEDAP.KL_LAMBDA ${KL_LAMBDA}
        fi
    done
done
