#!/usr/bin/env bash
# Pretrain 10 nnUNetv2 models using 2 GPUs (run 2 seeds in parallel).
#
# Prerequisite (already done per your note):
#   ./bash_script/prepare_nnunet_pretraining_data.sh
#
# Usage:
#   chmod +x bash_script/run_pretrain_10seeds_2gpus.sh
#   ./bash_script/run_pretrain_10seeds_2gpus.sh
#
# Optional overrides:
#   GPUS="0,1"              (default: 0,1)
#   SEEDS="1 2 ... 10"      (default: 1..10)
#   FOLD=0                  (default: 0)
#   PROJECT_ROOT=...         (default: pwd)
#   CONDA_ENV_PATH=...       (default: ${PROJECT_ROOT}/noisy_seg_env)
#
# Notes:
# - This starts two independent nnUNet trainings, each bound to one GPU via CUDA_VISIBLE_DEVICES.
# - Outputs go to: ${PROJECT_ROOT}/nnUNet_models/seed_<seed>/

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${PROJECT_ROOT}/noisy_seg_env}"

RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/nnUNet_raw}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${PROJECT_ROOT}/nnUNet_preprocessed}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/nnUNet_models}"

DATASET_ID="${DATASET_ID:-1}"
DATASET_NAME="${DATASET_NAME:-Dataset001_nnunet}"
CONFIG_NAME="${CONFIG_NAME:-3d_fullres}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
FOLD="${FOLD:-0}"

GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -lt 2 ]; then
  echo "Error: GPUS must contain two GPU ids, e.g. GPUS=\"0,1\" (got '${GPUS}')." >&2
  exit 2
fi

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"

PREPROC_DATA_DIR="${PREPROCESSED_ROOT}/${DATASET_NAME}/${PLANS_ID}_${CONFIG_NAME}"
PREPROC_PLANS_JSON="${PREPROCESSED_ROOT}/${DATASET_NAME}/${PLANS_ID}.json"
PREPROC_FINGERPRINT_JSON="${PREPROCESSED_ROOT}/${DATASET_NAME}/dataset_fingerprint.json"

cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}" || true
  fi
fi

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
  echo "Error: nnUNetv2_train is not on PATH. Activate your env first:" >&2
  echo "  conda activate \"${CONDA_ENV_PATH}\"" >&2
  exit 1
fi

if [ ! -d "${PREPROC_DATA_DIR}" ] || [ ! -f "${PREPROC_PLANS_JSON}" ] || [ ! -f "${PREPROC_FINGERPRINT_JSON}" ]; then
  echo "Error: nnUNet preprocessed data not found. Run:" >&2
  echo "  ./bash_script/prepare_nnunet_pretraining_data.sh" >&2
  echo "Expected: ${PREPROC_DATA_DIR}" >&2
  exit 1
fi

export nnUNet_raw="${RAW_ROOT}"
export nnUNet_preprocessed="${PREPROCESSED_ROOT}"

mkdir -p "${RESULTS_ROOT}"
LOG_ROOT="${RESULTS_ROOT}/train_logs"
mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "Pretraining nnUNetv2"
echo "  dataset_id:     ${DATASET_ID} (${DATASET_NAME})"
echo "  config:         ${CONFIG_NAME}"
echo "  fold:           ${FOLD}"
echo "  seeds:          ${SEEDS}"
echo "  gpus:           ${GPU_LIST[*]} (2-way parallel)"
echo "  results_root:   ${RESULTS_ROOT}"
echo "============================================================"

run_one_seed() {
  local seed="$1"
  local gpu="$2"
  local true_seed=$((833000 + seed))

  local out_root="${RESULTS_ROOT}/seed_${seed}"
  local log_dir="${LOG_ROOT}/seed_${seed}"
  mkdir -p "${out_root}" "${log_dir}"

  echo "[seed ${seed}] GPU=${gpu} nnUNet_random_seed=${true_seed}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export nnUNet_results="${out_root}"
    export nnUNet_random_seed="${true_seed}"
    nnUNetv2_train "${DATASET_ID}" "${CONFIG_NAME}" "${FOLD}"
  ) >"${log_dir}/train_stdout.log" 2>"${log_dir}/train_stderr.log"
}

max_jobs=2
declare -a pids=()
declare -a running_seeds=()

gpu_idx=0
for seed in ${SEEDS}; do
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Skip invalid seed '${seed}'" >&2
    continue
  fi

  # Wait until a slot is free
  while [ "${#pids[@]}" -ge "${max_jobs}" ]; do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}" || true
        unset 'pids[i]'
        unset 'running_seeds[i]'
        pids=("${pids[@]}")
        running_seeds=("${running_seeds[@]}")
        break
      fi
    done
    sleep 5
  done

  gpu="${GPU_LIST[$gpu_idx]}"
  gpu_idx=$(((gpu_idx + 1) % max_jobs))

  run_one_seed "${seed}" "${gpu}" &
  pids+=("$!")
  running_seeds+=("${seed}")
done

fail=0
for i in "${!pids[@]}"; do
  seed="${running_seeds[$i]}"
  if ! wait "${pids[$i]}"; then
    echo "[seed ${seed}] FAILED (see ${LOG_ROOT}/seed_${seed}/train_stderr.log)" >&2
    fail=1
  else
    echo "[seed ${seed}] OK"
  fi
done

if [ "${fail}" -ne 0 ]; then
  exit 1
fi

echo "All seeds finished."

