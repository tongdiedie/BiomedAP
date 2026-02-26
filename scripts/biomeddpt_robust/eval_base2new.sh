#!/bin/bash

# BiomedDPT_Robust Base-to-New Evaluation Script
# Usage: bash eval_base2new_robust.sh <DATA_PATH> <DATASET>
# Example: bash eval_base2new_robust.sh data BTMRI

DATA=$1
DATASET=$2
SHOTS=16
MODEL=BiomedCLIP
NCTX=4
CSC=False
CTP=end
LOADEP=50

METHOD=BiomedDPT_Robust
TRAINER=BiomedDPT_Robust_${MODEL}
SUB_base=base
SUB_novel=new

# 【新增】Robust 参数配置
LOW_TEMPLATE_TYPE=minimal
# L1_LAMBDA_HIGH=0.5
# L1_LAMBDA_LOW=0.3
# KL_LAMBDA=0.1

for SEED in 1 2 3
do
    MODEL_DIR=base2new_robust/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}_low${LOW_TEMPLATE_TYPE}/seed${SEED}
    if [ -d "$MODEL_DIR" ]; then
        echo "The checkpoint exists at ${MODEL_DIR} (skipping to evaluation)"
    else
        echo "Warning: Checkpoint not found at ${MODEL_DIR}"
        echo "Please train the model first using base2new_robust.sh"
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
                TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
                # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
                # TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
                # TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA ${KL_LAMBDA}

        fi
    done
done
