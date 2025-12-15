#!/bin/bash -l

#$ -P ec500kb
#$ -N nnunet_predict
#$ -l h_rt=47:00:00
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l mem_per_core=4G
#$ -j y
#$ -m ea

cd /projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM

export SEED_ID=10

conda activate conda_envs/nnunet

PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"

export nnUNet_raw="$PROJECT_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/data/nnUNet_results/Dataset001_nnunet/seed_${SEED_ID}"

mkdir -p "$PROJECT_ROOT/data/nnUNet_results/Dataset001_nnunet/preds/seed_${SEED_ID}"


nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset001_nnunet/imagesTs" \
  -o "$PROJECT_ROOT/data/nnUNet_results/Dataset001_nnunet/preds/seed_${SEED_ID}" \
  -d 001 -c 3d_fullres -p nnUNetPlans -tr nnUNetTrainer \
  -f 0 \
  --save_probabilities
