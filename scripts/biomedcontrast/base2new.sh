
#!/bin/bash
# ==================== base2new.sh ====================
# BiomedContrast Base-to-New Generalization Script
# Usage: bash base2new.sh DATA_PATH DATASET MODEL

DATA=$1
DATASET=$2
MODEL=$3
METHOD=BiomedContrast
TRAINER=BiomedContrast_${MODEL}

SHOTS=16
LOADEP=50  # Base2new任务加载epoch 50
CTP=end
CSC=False
NCTX=4
SUB_base=base
SUB_novel=new

for SEED in 1 2 3
do
    # ===== Stage 1: 在base类上训练 =====
    DIR=output/base2new/train_${SUB_base}/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "======================================"
        echo "Training on BASE classes - Seed ${SEED}"
        echo "Dataset: ${DATASET}"
        echo "Method: ${METHOD} with negative sample repulsion"
        echo "======================================"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${METHOD}/base_to_novel/${DATASET}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB_base}
    fi
    
    # ===== Stage 2: 在novel类上测试 =====
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    MODEL_DIR=output/base2new/train_${SUB_base}/${COMMON_DIR}
    DIR=output/base2new/test_${SUB_novel}/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "======================================"
        echo "Testing on NOVEL classes - Seed ${SEED}"
        echo "Loading model from: ${MODEL_DIR}"
        echo "======================================"
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
        DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
    fi
done

echo "======================================"
echo "Base-to-New training completed!"
echo "======================================"