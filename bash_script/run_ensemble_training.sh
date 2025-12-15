#!/bin/bash
set -euo pipefail

# Ensemble Training Script for 10 U-Net Models
# Uses process-based parallel training (optimized for L40S GPU with 32 CPU cores)
# 
# Usage:
#   ./scripts/run_ensemble_training.sh          - Start new training
#   ./scripts/run_ensemble_training.sh --resume - Resume from checkpoints
#   
# Check progress: ./scripts/manage_checkpoints.sh status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/noisy_seg_env/bin/python"

# Configuration - Optimized for L40S (48GB VRAM) + 32 CPU cores
NUM_MODELS=10
NUM_PARALLEL=4       # Models trained simultaneously (L40S has enough VRAM)
BATCH_SIZE=4         # Batch size per model 
NUM_WORKERS=8        # Data loading workers per model 
EPOCHS=100
LEARNING_RATE=0.0001
OUTPUT_DIR="${PROJECT_ROOT}/results/output_ensemble"
DATA_DIR="${PROJECT_ROOT}/data"
TRAIN_LIST="${PROJECT_ROOT}/data/splits/train_list.txt"

# Capture optional first argument safely for set -u
ARG1="${1-}"

# Check if models are already trained
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    # Count how many models have completed training
    COMPLETED=0
    for i in $(seq 0 $((NUM_MODELS - 1))); do
        MODEL_CHECKPOINT="$CHECKPOINT_DIR/model_$i/latest.pth"
        if [ -f "$MODEL_CHECKPOINT" ]; then
            # Check if this model has completed training
            EPOCH=$("${PYTHON_BIN}" -c "import torch; ckpt = torch.load('$MODEL_CHECKPOINT', map_location='cpu'); print(ckpt.get('epoch', 0))" 2>/dev/null || echo "0")
            if [ "$EPOCH" -ge "$EPOCHS" ]; then
                COMPLETED=$((COMPLETED + 1))
            fi
        fi
    done
    
    # If models are already trained and not resuming, ask for confirmation
    if [ $COMPLETED -gt 0 ] && [ "$ARG1" != "--resume" ] && [ "$ARG1" != "-r" ]; then
        echo "========================================"
        echo "WARNING: Trained models detected!"
        echo "========================================"
        echo "$COMPLETED out of $NUM_MODELS models have completed training."
        echo ""
        echo "Starting new training will reset all progress."
        echo ""
        read -p "Do you want to continue and OVERWRITE existing models? (yes/no): " -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            echo "Training cancelled."
            echo ""
            echo "To resume training instead, use:"
            echo "  ./scripts/run_ensemble_training.sh --resume"
            exit 0
        fi
        echo "Continuing with new training (existing models will be overwritten)..."
        echo ""
    fi
fi

# Add --resume flag to continue from checkpoints if they exist
RESUME_FLAG=""
if [ "$ARG1" == "--resume" ] || [ "$ARG1" == "-r" ]; then
    RESUME_FLAG="--resume"
    echo "========================================"
    echo "RESUMING ensemble training from checkpoints"
    echo "========================================"
fi

echo "========================================"
echo "Ensemble Training Configuration"
echo "========================================"
echo "Number of models: $NUM_MODELS"
echo "Parallel models: $NUM_PARALLEL"
echo "Batch size per model: $BATCH_SIZE"
echo "Data loading workers per model: $NUM_WORKERS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: ./results/output_ensemble"
echo "Data directory: ./data"
if [ -f "$TRAIN_LIST" ]; then
    echo "Train list: ./data/splits/train_list.txt"
else
    echo "Train list: (not found) ./data/splits/train_list.txt"
fi
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume mode: ENABLED"
fi
echo "========================================"
echo ""

# Verify train list existence
TRAIN_LIST_ARGS=()
if [ -f "$TRAIN_LIST" ]; then
    TRAIN_LIST_ARGS+=(--train_list "$TRAIN_LIST")
else
    echo "Warning: train list '$TRAIN_LIST' not found; falling back to all available samples."
    echo ""
fi

# Run ensemble training
CMD=(
    "${PYTHON_BIN}" -m model_script.train.train_ensemble
    --num_models "$NUM_MODELS"
    --num_parallel "$NUM_PARALLEL"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --output_dir "$OUTPUT_DIR"
    --data_dir "$DATA_DIR"
)

if [ ${#TRAIN_LIST_ARGS[@]} -gt 0 ]; then
    CMD+=("${TRAIN_LIST_ARGS[@]}")
fi

if [ -n "$RESUME_FLAG" ]; then
    CMD+=(--resume)
fi

# Respect manual device override via CUDA_VISIBLE_DEVICES
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    CMD+=(--devices "$CUDA_VISIBLE_DEVICES")
fi

"${CMD[@]}"

echo ""
echo "========================================"
echo "Ensemble training complete!"
echo "Models saved to: $OUTPUT_DIR/checkpoints/"
echo "Logs saved to: $OUTPUT_DIR/logs/"
echo "========================================"
