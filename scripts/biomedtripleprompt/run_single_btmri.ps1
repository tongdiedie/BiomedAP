# ========================================
# 单次运行脚本（快速测试）
# 文件位置：scripts/run_single_btmri.ps1
# ========================================

$env:CUDA_VISIBLE_DEVICES = "0"

python train.py `
    --root data `
    --trainer BiomedTriplePrompt `
    --dataset-config-file configs/datasets/btmri.yaml `
    --config-file configs/trainers/BiomedTriplePrompt/btmri.yaml `
    --output-dir output/test_btmri_triple `
    TRAINER.BIOMEDTRIPLEPROMPT.LOW_TEMPLATE_TYPE minimal `
    DATASET.NUM_SHOTS 16
