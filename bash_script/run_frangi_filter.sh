#!/bin/bash
set -euo pipefail

# Frangi Filter Processing Script
# Applies vesselness filter to enhance tubular structures in 3D medical images

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/noisy_seg_env/bin/python"
FRANGI_SCRIPT="${PROJECT_ROOT}/frangi_filter.py"

# Configuration
INPUT_DIR="${PROJECT_ROOT}/data/images"
OUTPUT_DIR="${PROJECT_ROOT}/data/frangi"
SIGMAS="1,2,3,4,5"  # Multi-scale filtering
ALPHA=0.5           # Plate-like structure sensitivity
BETA=0.5            # Blob-like structure sensitivity
GAMMA=15            # Background suppression
NUM_WORKERS=32      # Parallel workers (adjust based on your CPU)

echo "========================================"
echo "Frangi Filter Processing"
echo "========================================"
echo "Input:  ./data/images"
echo "Output: ./data/frangi"
echo "Workers: $NUM_WORKERS"
echo "Sigmas: $SIGMAS"
echo "========================================"
echo ""

# Run Frangi filter processing
"${PYTHON_BIN}" "${FRANGI_SCRIPT}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --sigmas "$SIGMAS" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "Done! Filtered images saved to $OUTPUT_DIR"
