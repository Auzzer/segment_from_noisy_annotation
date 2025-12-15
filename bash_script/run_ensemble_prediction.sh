#!/bin/bash
set -euo pipefail

# Ensemble Prediction Script for 10 U-Net Models
# Performs inference using trained ensemble models and generates predictions
#
# Usage:
#   ./scripts/run_ensemble_prediction.sh          - Start new prediction
#   ./scripts/run_ensemble_prediction.sh --resume - Resume from interrupted prediction

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/noisy_seg_env/bin/python"

# Configuration
NUM_MODELS=10
CHECKPOINT_DIR="${PROJECT_ROOT}/results/output_ensemble"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/results/predictions_ensemble"
PATCH_SIZE="64 64 64"
OVERLAP=0.5
THRESHOLD=0.5
PATCH_BATCH_SIZE=128
USE_AMP=false
TRAIN_LIST="${PROJECT_ROOT}/data/splits/train_list.txt"
TEST_LIST="${PROJECT_ROOT}/data/splits/test_list.txt"
SAVE_PER_MODEL_PREDS=false
PER_MODEL_OUTPUT_ROOT="${PROJECT_ROOT}/data/unet_prediction"
ENSEMBLE_OUTPUT_ROOT="${PROJECT_ROOT}/data/unet_prediction/ensemble"
BINARIZE_SAVED_OUTPUTS=true
SKIP_METRICS=true

# Check for resume flag (handle no-arg safely)
RESUME_FLAG=""
if [ "${1-}" == "--resume" ] || [ "${1-}" == "-r" ]; then
    RESUME_FLAG="--resume"
    echo "========================================"
    echo "RESUMING ensemble prediction"
    echo "========================================"
    echo ""
fi

echo "========================================"
echo "Ensemble Prediction"
echo "========================================"
echo "Number of models: $NUM_MODELS"
echo "Checkpoint directory: ./results/output_ensemble"
echo "Data directory: ./data"
echo "Output directory: ./results/predictions_ensemble"
echo "Patch size: $PATCH_SIZE"
echo "Overlap: $OVERLAP"
echo "Threshold: $THRESHOLD"
echo "Patch batch size: $PATCH_BATCH_SIZE"
echo "Use AMP: $USE_AMP"
echo "Save per-model predictions: $SAVE_PER_MODEL_PREDS"
if [ "$SAVE_PER_MODEL_PREDS" = true ]; then
    echo "Per-model output root: $PER_MODEL_OUTPUT_ROOT"
fi
echo "Ensemble output root: ${ENSEMBLE_OUTPUT_ROOT:-'(default output dir only)'}"
echo "Binarize saved outputs: $BINARIZE_SAVED_OUTPUTS"
echo "Skip metrics: $SKIP_METRICS"
echo "Train list: ${TRAIN_LIST:-'(not specified)'}"
echo "Test list: ${TEST_LIST:-'(not specified)'}"
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume mode: ENABLED"
fi
echo "========================================"
echo ""

# Verify list existence (train + test)
IMAGE_LIST_ARGS=()
if [ -n "$TRAIN_LIST" ]; then
    if [ -f "$TRAIN_LIST" ]; then
        IMAGE_LIST_ARGS+=(--image_list "$TRAIN_LIST")
    else
        echo "Warning: train list '$TRAIN_LIST' not found; will skip it."
    fi
fi

if [ -n "$TEST_LIST" ]; then
    if [ -f "$TEST_LIST" ]; then
        IMAGE_LIST_ARGS+=(--image_list "$TEST_LIST")
    else
        echo "Warning: test list '$TEST_LIST' not found; will skip it."
    fi
fi

# Run prediction
CMD=(
    "${PYTHON_BIN}" -m model_script.predict.predict_ensemble
    --checkpoint_dir "$CHECKPOINT_DIR"
    --num_models "$NUM_MODELS"
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --patch_size $PATCH_SIZE
    --overlap "$OVERLAP"
    --threshold "$THRESHOLD"
    --patch_batch_size "$PATCH_BATCH_SIZE"
)

if [ "$USE_AMP" = true ]; then
    CMD+=(--amp)
fi

if [ "$SAVE_PER_MODEL_PREDS" = true ]; then
    CMD+=(--save_per_model_preds --per_model_output_root "$PER_MODEL_OUTPUT_ROOT")
fi

if [ -n "$ENSEMBLE_OUTPUT_ROOT" ]; then
    CMD+=(--ensemble_output_root "$ENSEMBLE_OUTPUT_ROOT")
fi

if [ "$BINARIZE_SAVED_OUTPUTS" = true ]; then
    CMD+=(--binarize_saved_outputs)
fi

if [ "$SKIP_METRICS" = true ]; then
    CMD+=(--skip_metrics)
fi

if [ ${#IMAGE_LIST_ARGS[@]} -gt 0 ]; then
    CMD+=("${IMAGE_LIST_ARGS[@]}")
fi

if [ -n "$RESUME_FLAG" ]; then
    CMD+=(--resume)
fi

"${CMD[@]}"

echo ""
echo "========================================"
echo "Ensemble prediction complete!"
echo "Results saved to: ./results/predictions_ensemble"
echo "========================================"
