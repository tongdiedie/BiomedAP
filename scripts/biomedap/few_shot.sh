#!/bin/bash

# BiomedAP Few-shot Training Script
# Usage: bash few_shot.sh <DATA_PATH> <DATASET> <SHOTS> <MODEL>
# Example: bash few_shot.sh data BTMRI 16 BiomedCLIP

DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=$4  # BiomedCLIP, CLIP, PubMedCLIP, PMCCLIP
NCTX=4
CSC=False
CTP=end

# 【新增】BiomedAP 参数配置
LOW_TEMPLATE_TYPE=minimal  # 低质量模板类型: minimal, simple, basic
# L1_LAMBDA_HIGH=25.0      # 高质量对齐权重
# L1_LAMBDA_LOW=0.3        # 低质量鲁棒性权重
# KL_LAMBDA=0.5            # 知识蒸馏权重

METHOD=BiomedAP
TRAINER=BiomedAP_${MODEL}

export CUDA_VISIBLE_DEVICES=0

for SEED in 1 2 3
do
    DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
            --output-dir ${DIR} \
            TRAINER.BIOMEDAP.N_CTX ${NCTX} \
            TRAINER.BIOMEDAP.CSC ${CSC} \
            TRAINER.BIOMEDAP.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
            DATASET.NUM_SHOTS ${SHOTS}
            # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
            # TRAINER.BIOMEDAP.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
            # TRAINER.BIOMEDAP.KL_LAMBDA ${KL_LAMBDA}
    fi
done
