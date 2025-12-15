# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 17:44:06 2025

@author: jkwar
"""

#!/usr/bin/env python3
"""
STAPLE Ensemble Prediction Script
Runs nnUNet ensemble prediction and STAPLE fusion.

This script automates:
1. Running nnUNetv2_ensemble to combine predictions from seeds 1-5
2. Evaluating the ensemble with Dice metrics
3. Saving results to the specified output directory

Usage:
    python run_staple_ensemble.py

Requirements:
    - nnUNet conda environment must be activated
    - Predictions from seeds 1-5 must already exist
"""

import subprocess
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM")
PRED_ROOT = PROJECT_ROOT / "data/nnUNet_results/Dataset003_vesselfm_good/preds"
OUTPUT_ENSEMBLE = PRED_ROOT / "ensemble_seed1-5"
GT_DIR = PROJECT_ROOT / "data/nnUNet_raw/Dataset003_vesselfm_good/labelsTs"
PFILE = PROJECT_ROOT / "data/nnUNet_results/Dataset003_vesselfm_good/seed_1/Dataset003_vesselfm_good/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
DJFILE = PROJECT_ROOT / "data/nnUNet_raw/Dataset003_vesselfm_good/dataset.json"


def check_seed_predictions():
    """Verify that all required seed prediction folders exist."""
    print("[INFO] Checking seed prediction folders...")
    missing = []
    for seed in range(1, 6):
        seed_dir = PRED_ROOT / f"seed_{seed}"
        if not seed_dir.exists():
            missing.append(f"seed_{seed}")
    
    if missing:
        print(f"[ERROR] Missing prediction folders: {', '.join(missing)}")
        print(f"[ERROR] Expected location: {PRED_ROOT}/seed_N/")
        sys.exit(1)
    
    print("[OK] All seed_1..seed_5 folders found.")


def check_ground_truth():
    """Verify ground truth directory exists."""
    if not GT_DIR.exists():
        print(f"[ERROR] Ground-truth folder not found: {GT_DIR}")
        sys.exit(1)
    print(f"[OK] Ground-truth directory found.")


def run_ensemble():
    """Run nnUNetv2_ensemble to combine predictions."""
    print("\n" + "="*60)
    print("Running nnUNetv2_ensemble for STAPLE fusion...")
    print("="*60)
    
    # Create output directory
    OUTPUT_ENSEMBLE.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensemble output -> {OUTPUT_ENSEMBLE}")
    
    # Build ensemble command
    cmd = [
        "nnUNetv2_ensemble",
        "-i",
        str(PRED_ROOT / "seed_1"),
        str(PRED_ROOT / "seed_2"),
        str(PRED_ROOT / "seed_3"),
        str(PRED_ROOT / "seed_4"),
        str(PRED_ROOT / "seed_5"),
        "-o",
        str(OUTPUT_ENSEMBLE)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("[OK] STAPLE ensemble complete!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ensemble failed with exit code {e.returncode}")
        print(f"[ERROR] {e.stderr}")
        sys.exit(1)


def evaluate_ensemble():
    """Evaluate ensemble predictions and compute Dice scores."""
    print("\n" + "="*60)
    print("Evaluating Dice scores...")
    print("="*60)
    
    metrics_json = OUTPUT_ENSEMBLE / "metrics.json"
    print(f"[INFO] Metrics will be saved to: {metrics_json}")
    
    # Build evaluation command
    cmd = [
        "nnUNetv2_evaluate_folder",
        str(GT_DIR),
        str(OUTPUT_ENSEMBLE),
        "-djfile", str(DJFILE),
        "-pfile", str(PFILE),
        "-o", str(metrics_json)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("[OK] Evaluation finished.")
        print(f"[INFO] Results saved to: {metrics_json}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed with exit code {e.returncode}")
        print(f"[ERROR] {e.stderr}")
        sys.exit(1)


def main():
    """Main execution function."""
    print("="*60)
    print("STAPLE Ensemble Prediction Pipeline")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Working directory: {Path.cwd()}")
    print()
    
    # Step 1: Verify prerequisites
    check_seed_predictions()
    check_ground_truth()
    
    # Step 2: Run ensemble
    run_ensemble()
    
    # Step 3: Evaluate
    evaluate_ensemble()
    
    # Done
    print("\n" + "="*60)
    print("STAPLE ensemble prediction complete!")
    print("="*60)
    print(f"\nResults location: {OUTPUT_ENSEMBLE}")
    print(f"Metrics file: {OUTPUT_ENSEMBLE / 'metrics.json'}")
    print()
    print("Next steps:")
    print("  1. Run analyze_staple_results.py to generate visualizations")
    print("  2. Check metrics.json for detailed per-case results")
    print("="*60)


if __name__ == "__main__":
    main()