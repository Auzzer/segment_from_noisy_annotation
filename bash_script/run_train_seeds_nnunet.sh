#!/usr/bin/env bash
# Train nnUNetv2 models for multiple seeds (local terminal).
#
# What this script does:
#   1) Builds nnUNet_raw/Dataset001_nnunet from ./data/images + ./data/labels
#      using CT normalization: clip HU to [-1000, 400], then min-max to [0, 1].
#   2) Runs nnUNetv2_plan_and_preprocess once (seed 1), unless already done.
#   3) Trains one seed (per run) and syncs results back to nnUNet_models/seed_X.
#
# Usage:
#   chmod +x bash_script/run_train_seeds_nnunet.sh
#   ./bash_script/run_train_seeds_nnunet.sh 1
#   ./bash_script/run_train_seeds_nnunet.sh 2
#   ...
#   ./bash_script/run_train_seeds_nnunet.sh 10

set -euo pipefail

# ---- Seed selection ----
RUN_ID="${RUN_ID:-$$}"
SEED_ID="${1:-1}"
if ! [[ "${SEED_ID}" =~ ^[0-9]+$ ]]; then
    echo "Error: SEED_ID must be an integer (got '${SEED_ID}')." >&2
    exit 2
fi
TRUE_SEED=$((833000 + SEED_ID))

# ---- Project paths (override via env vars if needed) ----
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${PROJECT_ROOT}/noisy_seg_env}"

RAW_ROOT="${PROJECT_ROOT}/nnUNet_raw"
PREPROCESSED_ROOT="${PROJECT_ROOT}/nnUNet_preprocessed"
RESULTS_ROOT="${PROJECT_ROOT}/nnUNet_models"

DATA_IMAGES="${PROJECT_ROOT}/data/images"
DATA_LABELS="${PROJECT_ROOT}/data/labels"

DATASET_NAME="Dataset001_nnunet"
DATASET_ID=1
PLANS_ID="nnUNetPlans"
CONFIG_NAME="3d_fullres"

PREPROC_DATA_DIR="${PREPROCESSED_ROOT}/${DATASET_NAME}/${PLANS_ID}_${CONFIG_NAME}"
PREPROC_PLANS_JSON="${PREPROCESSED_ROOT}/${DATASET_NAME}/${PLANS_ID}.json"
PREPROC_FINGERPRINT_JSON="${PREPROCESSED_ROOT}/${DATASET_NAME}/dataset_fingerprint.json"

READY_FLAG="${PROJECT_ROOT}/.nnunet_preprocessed_ready"

SCRATCH_BASE="${SCRATCH_BASE:-/scratch}"
if [ ! -d "${SCRATCH_BASE}" ] || [ ! -w "${SCRATCH_BASE}" ]; then
    SCRATCH_BASE="${PROJECT_ROOT}/scratch"
    mkdir -p "${SCRATCH_BASE}"
fi
SCRATCH_ROOT="${SCRATCH_BASE}/nnunet_${RUN_ID}_seed${SEED_ID}"

echo "========================================"
echo "nnUNet training (seed ${SEED_ID})"
echo "Project root:   ${PROJECT_ROOT}"
echo "Scratch root:   ${SCRATCH_ROOT}"
echo "Results target: ${RESULTS_ROOT}/seed_${SEED_ID}"
echo "========================================"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV_PATH}" || true
    fi
fi

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1 || ! command -v nnUNetv2_train >/dev/null 2>&1; then
    echo "Error: nnUNetv2 is not on PATH. Activate your env first:" >&2
    echo "  conda activate \"${CONDA_ENV_PATH}\"" >&2
    exit 1
fi

mkdir -p "${SCRATCH_ROOT}" "${SCRATCH_ROOT}/nnUNet_results/seed_${SEED_ID}"

preprocess_finished() {
    [ -d "${PREPROC_DATA_DIR}" ] && [ -f "${PREPROC_PLANS_JSON}" ] && [ -f "${PREPROC_FINGERPRINT_JSON}" ]
}

# If preprocessing already exists but the ready flag is missing, create it.
if preprocess_finished && [ ! -f "${READY_FLAG}" ]; then
    touch "${READY_FLAG}"
fi

if [ "${SEED_ID}" -eq 1 ]; then
    if preprocess_finished; then
        echo "[Seed 1] Preprocessing already finished at ${PREPROC_DATA_DIR}; skipping plan+preprocess."
        touch "${READY_FLAG}"
    else
        echo "[Seed 1] Preparing nnUNet_raw (${DATASET_NAME}) from ${DATA_IMAGES} + ${DATA_LABELS} ..."
        export nnUNet_raw="${RAW_ROOT}"
        export nnUNet_preprocessed="${PREPROCESSED_ROOT}"
        export nnUNet_results="${RESULTS_ROOT}"

        python - <<'PY'
import json
from pathlib import Path

import nibabel as nib
import numpy as np

project_root = Path.cwd()
images_dir = project_root / "data/images"
labels_dir = project_root / "data/labels"

raw_dir = project_root / "nnUNet_raw" / "Dataset001_nnunet"
img_out = raw_dir / "imagesTr"
lbl_out = raw_dir / "labelsTr"
img_out.mkdir(parents=True, exist_ok=True)
lbl_out.mkdir(parents=True, exist_ok=True)

def ct_norm(vol: np.ndarray) -> np.ndarray:
    clipped = np.clip(vol, -1000.0, 400.0)
    return ((clipped + 1000.0) / 1400.0).astype(np.float32, copy=False)

image_files = sorted([p for p in images_dir.glob("image_*.nii*")])
if not image_files:
    raise SystemExit(f"No images found in {images_dir}")

label_values = set()
num_cases = 0

for img_path in image_files:
    case_id = img_path.stem.replace("image_", "").split(".")[0]
    lbl_path = labels_dir / f"label_{case_id}.nii.gz"
    if not lbl_path.exists():
        lbl_path = labels_dir / f"label_{case_id}.nii"
    if not lbl_path.exists():
        print(f"Skip {case_id}: label not found")
        continue

    img = nib.load(str(img_path))
    lbl = nib.load(str(lbl_path))

    img_data = ct_norm(img.get_fdata().astype(np.float32, copy=False))
    lbl_data = lbl.get_fdata().astype(np.uint8, copy=False)
    label_values |= set(np.unique(lbl_data).tolist())

    # nnUNet naming: <case>_0000.nii.gz for images, <case>.nii.gz for labels
    case_name = f"image_{case_id}"
    nib.save(
        nib.Nifti1Image(img_data, img.affine, img.header),
        str(img_out / f"{case_name}_0000.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(lbl_data, lbl.affine, lbl.header),
        str(lbl_out / f"{case_name}.nii.gz"),
    )
    num_cases += 1
    print(f"Prepared case {case_name}")

if num_cases == 0:
    raise SystemExit("No training cases prepared (did labels match images?).")

labels = {"background": 0}
for v in sorted([int(v) for v in label_values if int(v) != 0]):
    labels["vessel" if v == 1 else f"label_{v}"] = v

# Prefer nnUNet's generator if available; fallback to writing a minimal dataset.json.
try:
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json  # type: ignore

    generate_dataset_json(
        output_folder=str(raw_dir),
        # We pre-normalize (clip/minmax) here, so tell nnUNet not to normalize again.
        channel_names={0: "noNorm"},
        labels=labels,
        num_training_cases=num_cases,
        file_ending=".nii.gz",
        dataset_name="Dataset001_nnunet",
    )
except Exception:
    dataset_json = {
        "channel_names": {"0": "noNorm"},
        "labels": labels,
        "numTraining": int(num_cases),
        "file_ending": ".nii.gz",
        "name": "Dataset001_nnunet",
    }
    with open(raw_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

print(f"Raw dataset ready at: {raw_dir}")
PY

        echo "[Seed 1] Running nnUNetv2_plan_and_preprocess..."
        nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c "${CONFIG_NAME}" --verify_dataset_integrity
        touch "${READY_FLAG}"
    fi
else
    echo "[Seed ${SEED_ID}] Waiting for preprocessing to be ready..."
    while [ ! -f "${READY_FLAG}" ]; do sleep 60; done
fi

echo "[Seed ${SEED_ID}] Syncing preprocessed data to scratch..."
rsync -a "${PREPROCESSED_ROOT}/" "${SCRATCH_ROOT}/nnUNet_preprocessed/"

export nnUNet_raw="${RAW_ROOT}"
export nnUNet_preprocessed="${SCRATCH_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${SCRATCH_ROOT}/nnUNet_results/seed_${SEED_ID}"
export nnUNet_random_seed="${TRUE_SEED}"

echo "[Seed ${SEED_ID}] Starting nnUNetv2_train..."
nnUNetv2_train "${DATASET_ID}" "${CONFIG_NAME}" 0

echo "[Seed ${SEED_ID}] Syncing checkpoints back to nnUNet_models..."
mkdir -p "${RESULTS_ROOT}/seed_${SEED_ID}"
rsync -a "${SCRATCH_ROOT}/nnUNet_results/seed_${SEED_ID}/" "${RESULTS_ROOT}/seed_${SEED_ID}/"

if [[ -n "${SCRATCH_ROOT}" && -d "${SCRATCH_ROOT}" && "${SCRATCH_ROOT}" == "/scratch/nnunet_${JOB_ID}_seed${SEED_ID}" ]]; then
    rm -rf -- "${SCRATCH_ROOT}"
    echo "[Seed ${SEED_ID}] Scratch cleaned."
fi

echo "[Seed ${SEED_ID}] Done."
