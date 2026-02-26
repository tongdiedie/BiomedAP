#!/bin/bash

# Quick Test Script for BiomedAP
# Usage: bash quick_test_fewshot.sh

DATA=data
DATASET=BTMRI
SHOTS=16
MODEL=BiomedCLIP
SEED=1

LOW_TEMPLATE_TYPE=minimal
# L1_LAMBDA_HIGH=25.0
# L1_LAMBDA_LOW=0.3
# KL_LAMBDA=0.5

METHOD=BiomedAP
TRAINER=BiomedAP_${MODEL}

DIR=output/quick_test_biomedap

echo "========================================" 
echo "   BiomedAP Quick Test (5 epochs)"
echo "========================================"
echo "Dataset:           ${DATASET}"
echo "Backbone:          ${MODEL}"
echo "Shots:             ${SHOTS}"
echo "Low Template:      ${LOW_TEMPLATE_TYPE}"
echo "========================================"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
    --output-dir ${DIR} \
    TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE ${LOW_TEMPLATE_TYPE} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH 5
    # TRAINER.BIOMEDAP.L1_LAMBDA_HIGH ${L1_LAMBDA_HIGH} \
    # TRAINER.BIOMEDAP.L1_LAMBDA_LOW ${L1_LAMBDA_LOW} \
    # TRAINER.BIOMEDAP.KL_LAMBDA ${KL_LAMBDA}

echo "Quick test completed! Results saved to ${DIR}"
