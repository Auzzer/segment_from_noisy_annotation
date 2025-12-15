#!/bin/bash
set -euo pipefail

# Single U-Net Training Script
# Trains a single 3D U-Net model for medical image segmentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/noisy_seg_env/bin/python"

echo "========================================"
echo "Single U-Net Training"
echo "========================================"
echo "Data directory: ./data"
echo "Output directory: ./results/single_model"
echo "Batch size: 2"
echo "Epochs: 100"
echo "Learning rate: 1e-4"
echo "Patch size: 64 64 64"
echo "Validation split: 0.2"
echo "Using: GPU (CUDA enabled)"
echo "========================================"
echo ""

# Run training with GPU enabled
"${PYTHON_BIN}" -m model_script.train.train \
    --data_dir "${PROJECT_ROOT}/data" \
    --output_dir "${PROJECT_ROOT}/results/single_model" \
    --batch_size 2 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --patch_size 64 64 64 \
    --val_split 0.2 \
    --num_workers 4 \
    --save_frequency 10

echo ""
echo "========================================"
echo "Training complete!"
echo "Models saved to: ./results/single_model/checkpoints/"
echo "Logs saved to: ./results/single_model/logs/"
echo "========================================"
