#!/bin/bash

# BiomedDPT_Robust Few-shot Training Script
# Usage: bash few_shot_robust.sh <DATA_PATH> <DATASET> <SHOTS> <MODEL>
# Example: bash few_shot_robust.sh data BTMRI 16 BiomedCLIP

DATA=$1
DATASET=$2
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=$4  # BiomedCLIP, CLIP, PubMedCLIP, PMCCLIP
NCTX=4
CSC=False
CTP=end

# 【新增】Robust 参数配置
LOW_TEMPLATE_TYPE=minimal  # 低质量模板类型: minimal, class_only, empty
# L1_LAMBDA_HIGH=0.5         
# L1_LAMBDA_LOW=0.3          
# KL_LAMBDA=0.1              

METHOD=BiomedDPT_Robust
TRAINER=BiomedDPT_Robust_${MODEL}

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
            TRAINER.BIOMEDDPT_ROBUST.N_CTX ${NCTX} \
            TRAINER.BIOMEDDPT_ROBUST.CSC ${CSC} \
            TRAINER.BIOMEDDPT_ROBUST.CLASS_TOKEN_POSITION ${CTP} \
            TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
            # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
            # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
            # TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA ${KL_LAMBDA} \
            DATASET.NUM_SHOTS ${SHOTS}
    fi
done
