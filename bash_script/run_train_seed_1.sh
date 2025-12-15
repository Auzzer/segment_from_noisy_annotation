#!/bin/bash -l

#$ -P ec500kb
#$ -N nnunet_train_702785
#$ -l h_rt=47:00:00
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l mem_per_core=4G
#$ -j y
#$ -m ea

cd /projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM

conda activate conda_envs/nnunet

export SEED_ID=1
export TRUE_SEED=833017

export PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"
export SCRATCH_ROOT="/scratch/nnunet_${JOB_ID}_seed${SEED_ID}"


mkdir -p "$SCRATCH_ROOT" "$SCRATCH_ROOT/nnUNet_results/seed_${SEED_ID}"
df -h /scratch | sed -n '1,2p'

rsync -a "$PROJECT_ROOT/data/nnUNet_preprocessed/" "$SCRATCH_ROOT/nnUNet_preprocessed/"

export nnUNet_preprocessed="${SCRATCH_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${SCRATCH_ROOT}/nnUNet_results/seed_${SEED_ID}"
export nnUNet_random_seed="${TRUE_SEED}"


echo "Scratch + env set. Starting training…" 

nnUNetv2_train Dataset001_nnunet 3d_fullres 0

mkdir -p "$PROJECT_ROOT/data/nnUNet_results/Dataset001_nnunet/seed_${SEED_ID}"
rsync -a "${SCRATCH_ROOT}/nnUNet_results/seed_${SEED_ID}/" "${PROJECT_ROOT}/data/nnUNet_results/Dataset001_nnunet/seed_${SEED_ID}/"

if [[ -n "${SCRATCH_ROOT:-}" && -d "$SCRATCH_ROOT" && "$SCRATCH_ROOT" == "/scratch/nnunet_${JOB_ID}_seed${SEED_ID}" ]]; then
    rm -rf -- "$SCRATCH_ROOT"
    echo "Scratch cleaned: $SCRATCH_ROOT"
else
    echo "Skip cleanup (not matching expected path)"
fi
