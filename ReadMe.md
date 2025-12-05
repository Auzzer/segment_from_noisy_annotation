# Noisy Segment — 3D Medical Image Segmentation

A practical pipeline for training, evaluating, and deploying 3D U-Net models (single or ensemble) on volumetric medical images.

## Features
- Single-model and ensemble training
- GPU-accelerated training and inference by default
- Safe checkpoint handling and resume support
- Sliding‑window inference with configurable patch size and overlap
- Built‑in metrics (Dice, clDice) and plotting utilities

## Project Layout
- `utils/` — reusable building blocks (`dataset.py`, `losses.py`, `unet_model.py`)
- `model_script/train/` — training drivers (`train.py`, `train_ensemble.py`, `check_progress.py`)
- `model_script/predict/` — inference/evaluation (`predict_ensemble.py`, `evaluate_predictions.py`)
- `model_script/preproc/` — data prep helpers (`generate_data_split.py`, ROI/frangi tools)
- `model_script/bo/` — Bayesian optimization drivers (e.g., Frangi parameter search)
- `scripts/` — shell entrypoints for train/predict/maintenance
- `visualization/` — plotting utilities
- `results/` — outputs (checkpoints, predictions, logs)

## Prerequisites
- Local Python environment at `./noisy_seg_env` (used by the scripts)
- Data arranged as below
- Make shell scripts executable (one-time):
  ```
  chmod +x scripts/*.sh
  ```

## Data Layout
```
data/
├── images/              # NIfTI images (*.nii.gz)
├── labels/              # NIfTI labels (*.nii.gz)
└── splits/
    ├── train_list.txt   # e.g., image_001, image_002, ...
    └── test_list.txt
```
Generate splits (optional, if you don’t have them yet):
```
./noisy_seg_env/bin/python -m model_script.preproc.generate_data_split --data_dir ./data
```

## Quick Start
1. Train an ensemble (default GPU):
   ```
   ./scripts/run_ensemble_training.sh
   ```
2. Check progress at any time:
   ```
   ./noisy_seg_env/bin/python -m model_script.train.check_progress
   ```
3. Run prediction (test list is used by the script):
   ```
   ./scripts/run_ensemble_prediction.sh
   ```
4. Evaluate and plot:
   ```
   ./noisy_seg_env/bin/python -m model_script.predict.evaluate_predictions
   ./noisy_seg_env/bin/python visualization/plot_results.py
   ```

## Training
### Single Model
- Start training:
  ```
  ./scripts/run_training.sh
  ```
- Resume from latest checkpoint:
  ```
  ./scripts/run_training.sh --resume
  ```
- Key knobs (via `model_script/train/train.py`): `--epochs`, `--batch_size`, `--learning_rate`, `--patch_size D H W`, `--num_workers`, `--val_split`, `--train_list`

### Ensemble
- Start/Resume:
  ```
  ./scripts/run_ensemble_training.sh
  ./scripts/run_ensemble_training.sh --resume
  ```
- Safety: When trained models are detected, you’ll be asked before overwriting. Use `--resume` to continue without prompts.

## Prediction (Ensemble)
- Standard run:
  ```
  ./scripts/run_ensemble_prediction.sh
  ```
- Resume interrupted prediction (skips already processed images):
  ```
  ./scripts/run_ensemble_prediction.sh --resume
  ```
- Optional fusion: `--fusion_method staple` to combine per-model masks via SimpleITK STAPLE (defaults to probability mean; requires `SimpleITK` installed).
- Useful toggles (via script/env): patch size, overlap, `PATCH_BATCH_SIZE`, and AMP for memory/perf.

## Checkpoints & Resuming
- Checkpoints saved per model (e.g., `latest.pth`, `best.pth`).
- Resume training with `--resume` on the respective script.
- Manage checkpoints:
  ```
  ./scripts/manage_checkpoints.sh status
  ./scripts/manage_checkpoints.sh backup
  ./scripts/manage_checkpoints.sh restore
  ```

## Evaluation & Plots
- Metrics over predictions:
  ```
  ./noisy_seg_env/bin/python -m model_script.predict.evaluate_predictions
  ```
- Visual summaries:
  ```
  ./noisy_seg_env/bin/python visualization/plot_results.py
  ./noisy_seg_env/bin/python visualization/plot_ensemble_lines.py
  ```


## Command Cheat Sheet
```
# Data
./noisy_seg_env/bin/python -m model_script.preproc.generate_data_split --data_dir ./data

# Training
./scripts/run_training.sh                         # Single model
./scripts/run_training.sh --resume
./scripts/run_ensemble_training.sh                # Ensemble
./scripts/run_ensemble_training.sh --resume

# Prediction
./scripts/run_ensemble_prediction.sh
./scripts/run_ensemble_prediction.sh --resume

# Checkpoints
./scripts/manage_checkpoints.sh status | backup | restore

# Evaluation & Plots
./noisy_seg_env/bin/python -m model_script.predict.evaluate_predictions
./noisy_seg_env/bin/python visualization/plot_results.py
./noisy_seg_env/bin/python visualization/plot_ensemble_lines.py
```
