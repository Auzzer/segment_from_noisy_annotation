#!/usr/bin/env bash
# Prepare nnUNetv2 "pretraining" data (nnUNet_raw + nnUNet_preprocessed) from:
#   ./data/images/image_XXX.nii.gz and ./data/labels/label_XXX.nii.gz
#
# CT normalization (fixed):
#   clip HU to [-1000, 400], then min-max to [0, 1] via (x+1000)/1400.
#
# This script is idempotent:
#   - If nnUNet_preprocessed/<Dataset>/nnUNetPlans_3d_fullres exists with plans+fingerprint,
#     it skips plan+preprocess.
#
# Usage:
#   chmod +x bash_script/prepare_nnunet_pretraining_data.sh
#   ./bash_script/prepare_nnunet_pretraining_data.sh
#
# Optional overrides (env vars):
#   PROJECT_ROOT, CONDA_ENV_PATH, DATASET_ID, DATASET_NAME, CONFIG_NAME
#   BINARIZE_LABELS=1    (map all non-zero label ids -> 1)
#   FORCE_CLEAN=1        (re-run plan+preprocess with --clean even if it exists)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${PROJECT_ROOT}/noisy_seg_env}"

RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/nnUNet_raw}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${PROJECT_ROOT}/nnUNet_preprocessed}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/nnUNet_models}"

DATA_IMAGES="${DATA_IMAGES:-${PROJECT_ROOT}/data/images}"
DATA_LABELS="${DATA_LABELS:-${PROJECT_ROOT}/data/labels}"

DATASET_ID="${DATASET_ID:-1}"
DATASET_NAME="${DATASET_NAME:-Dataset001_nnunet}"
CONFIG_NAME="${CONFIG_NAME:-3d_fullres}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
BINARIZE_LABELS="${BINARIZE_LABELS:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"

PREPROC_DATA_DIR="${PREPROCESSED_ROOT}/${DATASET_NAME}/${PLANS_ID}_${CONFIG_NAME}"
PREPROC_PLANS_JSON="${PREPROCESSED_ROOT}/${DATASET_NAME}/${PLANS_ID}.json"
PREPROC_FINGERPRINT_JSON="${PREPROCESSED_ROOT}/${DATASET_NAME}/dataset_fingerprint.json"
READY_FLAG="${PROJECT_ROOT}/.nnunet_preprocessed_ready"

preprocess_finished() {
    [ -d "${PREPROC_DATA_DIR}" ] && [ -f "${PREPROC_PLANS_JSON}" ] && [ -f "${PREPROC_FINGERPRINT_JSON}" ]
}

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV_PATH}" || true
    fi
fi

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1; then
    echo "Error: nnUNetv2 is not on PATH. Activate your env first:" >&2
    echo "  conda activate \"${CONDA_ENV_PATH}\"" >&2
    exit 1
fi

if preprocess_finished; then
    if [ "${FORCE_CLEAN}" != "1" ]; then
        echo "nnUNet preprocessing already finished:"
        echo "  ${PREPROC_DATA_DIR}"
        touch "${READY_FLAG}"
        exit 0
    fi
fi

export nnUNet_raw="${RAW_ROOT}"
export nnUNet_preprocessed="${PREPROCESSED_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"

echo "Preparing nnUNet_raw/${DATASET_NAME} from:"
echo "  images: ${DATA_IMAGES}"
echo "  labels: ${DATA_LABELS}"
echo "  binarize_labels: ${BINARIZE_LABELS}"

python - <<PY
import json
from pathlib import Path

import nibabel as nib
import numpy as np

project_root = Path("${PROJECT_ROOT}")
images_dir = Path("${DATA_IMAGES}")
labels_dir = Path("${DATA_LABELS}")
dataset_name = "${DATASET_NAME}"
binarize_labels = str("${BINARIZE_LABELS}").strip() == "1"

raw_dir = Path("${RAW_ROOT}") / dataset_name
img_out = raw_dir / "imagesTr"
lbl_out = raw_dir / "labelsTr"
img_out.mkdir(parents=True, exist_ok=True)
lbl_out.mkdir(parents=True, exist_ok=True)

def ct_norm(vol: np.ndarray) -> np.ndarray:
    clipped = np.clip(vol, -1000.0, 400.0)
    return ((clipped + 1000.0) / 1400.0).astype(np.float32, copy=False)

image_files = sorted(images_dir.glob("image_*.nii*"))
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
    if binarize_labels:
        lbl_data = (lbl_data > 0).astype(np.uint8, copy=False)
    label_values |= set(np.unique(lbl_data).tolist())

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

if binarize_labels:
    labels = {"background": 0, "vessel": 1}
else:
    labels = {"background": 0}
    for v in sorted([int(v) for v in label_values if int(v) != 0]):
        labels["vessel" if v == 1 else f"label_{v}"] = v

try:
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json  # type: ignore

    generate_dataset_json(
        output_folder=str(raw_dir),
        channel_names={0: "noNorm"},
        labels=labels,
        num_training_cases=num_cases,
        file_ending=".nii.gz",
        dataset_name=dataset_name,
    )
except Exception:
    dataset_json = {
        "channel_names": {"0": "noNorm"},
        "labels": labels,
        "numTraining": int(num_cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
    }
    with open(raw_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

print(f"Raw dataset ready at: {raw_dir}")
PY

echo "Running nnUNetv2_plan_and_preprocess (dataset_id=${DATASET_ID}, config=${CONFIG_NAME}) ..."
pp_args=()
if [ "${FORCE_CLEAN}" = "1" ]; then
    pp_args+=(--clean)
fi
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c "${CONFIG_NAME}" --verify_dataset_integrity "${pp_args[@]}"

touch "${READY_FLAG}"
echo "Done. Preprocessed data:"
echo "  ${PREPROC_DATA_DIR}"
