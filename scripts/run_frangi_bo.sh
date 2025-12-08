#!/bin/bash
set -euo pipefail

# Frangi BO hyperparameter search with two seeds (0 and 1)
# Usage: bash scripts/run_frangi_bo.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "CONDA_PREFIX is not set. Please activate your conda environment first." >&2
    exit 1
fi

# CUDA/toolchain paths based on the active conda env
export CUDA_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"

# Data/config paths (update if your dataset lives elsewhere)
TRAIN_LIST="/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/splits/train_list.txt"
ROI_IMAGES_DIR="/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/images"
ROI_LABELS_DIR="/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/labels"
WORK_DIR="${PROJECT_ROOT}/results/output_frangi_bo_seed0"
FRANGI_INPUT_DIR="${PROJECT_ROOT}/data/images"
FRANGI_OUTPUT_DIR="${PROJECT_ROOT}/data/frangi"

# BO/search settings
SIGMAS="0.6,0.8,1.2,1.6,2.0"
BOUNDS_ALPHA="0.45,0.55"
BOUNDS_BETA="0.45,0.55"
BOUNDS_GAMMA="3.0,8.0"
BOUNDS_THR="0.05,0.20"
N_INIT=12
N_ITER=40
BATCH_SIZE=3
DEVICE=0
THETA_WORKERS=1

COMMON_ARGS=(
    --train_list "$TRAIN_LIST"
    --roi_images_dir "$ROI_IMAGES_DIR"
    --roi_labels_dir "$ROI_LABELS_DIR"
    --work_dir "$WORK_DIR"
    --sigmas "$SIGMAS"
    --bounds_alpha "$BOUNDS_ALPHA"
    --bounds_beta "$BOUNDS_BETA"
    --bounds_gamma "$BOUNDS_GAMMA"
    --bounds_thr "$BOUNDS_THR"
    --n_init "$N_INIT"
    --n_iter "$N_ITER"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
    --theta_workers "$THETA_WORKERS"
    --no_preload
)

echo "Running Frangi BO with seeds 0 and 1..."
for SEED in 0 1; do
    echo "----------------------------------------"
    echo "Seed: $SEED"
    "$PYTHON_BIN" -m model_script.bo.frangi_BO "${COMMON_ARGS[@]}" --seed "$SEED"
done

echo "Done. BO results in: $WORK_DIR"

BEST_JSON="${WORK_DIR}/best.json"
if [ ! -f "$BEST_JSON" ]; then
    echo "No best.json found at $BEST_JSON. Skipping full-image filtering." >&2
    exit 1
fi

echo "Extracting best params from $BEST_JSON..."
read -r ALPHA BETA GAMMA THR <<<"$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
path = Path("${BEST_JSON}")
data = json.loads(path.read_text())
theta = data.get("theta")
if not theta or len(theta) != 4:
    raise SystemExit("Invalid 'theta' in best.json")
print(" ".join(str(x) for x in theta))
PY
)"
echo "Best theta: alpha=${ALPHA}, beta=${BETA}, gamma=${GAMMA}, thr=${THR}"

mkdir -p "$FRANGI_OUTPUT_DIR"
shopt -s nullglob
IMAGES=("$FRANGI_INPUT_DIR"/image_*.nii.gz)
if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "No input images found in $FRANGI_INPUT_DIR" >&2
    exit 1
fi

echo "Running Frangi filtering on ${#IMAGES[@]} images from $FRANGI_INPUT_DIR -> $FRANGI_OUTPUT_DIR"
for IMG in "${IMAGES[@]}"; do
    BASE=$(basename "$IMG")
    echo "  -> $BASE"
    "$PYTHON_BIN" -m model_script.bo.frangi_gpu \
        --input "$IMG" \
        --sigmas "$SIGMAS" \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --gamma "$GAMMA" \
        --threshold "$THR" \
        --device "$DEVICE" \
        --save_outputs \
        --output_dir "$FRANGI_OUTPUT_DIR"
done

echo "All images filtered. Outputs saved to $FRANGI_OUTPUT_DIR"
