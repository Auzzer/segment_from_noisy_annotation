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
PATCH_BATCH_SIZE=64
USE_AMP=false
TEST_LIST="${PROJECT_ROOT}/data/splits/test_list.txt"

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
if [ -n "$TEST_LIST" ]; then
    echo "Test list: ./data/splits/test_list.txt"
else
    echo "Test list: (not specified) processing all images"
fi
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume mode: ENABLED"
fi
echo "========================================"
echo ""

# Verify test list existence
TEST_LIST_ARGS=()
if [ -n "$TEST_LIST" ] && [ -f "$TEST_LIST" ]; then
    TEST_LIST_ARGS+=(--image_list "$TEST_LIST")
elif [ -n "$TEST_LIST" ]; then
    echo "Warning: test list '$TEST_LIST' not found; processing all images in data directory."
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

if [ ${#TEST_LIST_ARGS[@]} -gt 0 ]; then
    CMD+=("${TEST_LIST_ARGS[@]}")
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
