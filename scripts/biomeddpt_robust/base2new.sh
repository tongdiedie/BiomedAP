#!/bin/bash

# BiomedDPT_Robust Base-to-New Training and Evaluation Script
# Usage: bash base2new_robust.sh <DATA_PATH> <DATASET> <MODEL>
# Example: bash base2new_robust.sh data BTMRI BiomedCLIP

DATA=$1
DATASET=$2
MODEL=$3
METHOD=BiomedDPT_Robust
TRAINER=BiomedDPT_Robust_${MODEL}

SHOTS=16
LOADEP=50
CTP=end
CSC=False
NCTX=4
SUB_base=base
SUB_novel=new

# 【新增】Robust 参数配置
LOW_TEMPLATE_TYPE=minimal
# L1_LAMBDA_HIGH=0.5
# L1_LAMBDA_LOW=0.3
# KL_LAMBDA=0.1

for SEED in 1 2 3
do
    # ========== 阶段1：在 base classes 上训练 ==========
    DIR=output/base2new/train_${SUB_base}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB_base} \
            TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
            # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH $L1_LAMBDA_HIGH \
            # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW $L1_LAMBDA_LOW \
            # TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA $KL_LAMBDA

    fi
    
    # ========== 阶段2：在 new classes 上评估 ==========
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    MODEL_DIR=output/base2new/train_${SUB_base}/${COMMON_DIR}
    DIR=output/base2new/test_${SUB_novel}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB_novel} \
            TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
            # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
            # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
            # TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA ${KL_LAMBDA}
    fi
done
