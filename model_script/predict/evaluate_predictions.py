"""Evaluate saved ensemble predictions against ground-truth labels."""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

from .predict_ensemble import calculate_metrics


def evaluate_predictions(pred_dir: Path, labels_dir: Path, threshold: float, csv_path: Path | None):
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        pred_dir: Directory containing pred_XXX.nii.gz files
        labels_dir: Directory containing label_XXX.nii.gz files
        threshold: Threshold for binarizing predictions
        csv_path: Optional path to save per-image metrics CSV
    """
    pred_files = sorted(pred_dir.glob('pred_*.nii.gz'))
    if not pred_files:
        print(f"No prediction files found in {pred_dir} (expected pred_XXX.nii.gz).")
        return

    rows = []
    for pred_path in pred_files:
        try:
            basename = Path(pred_path.stem).stem  # handle .nii.gz double extension
            img_num = basename.split('_')[1]
        except IndexError:
            print(f"Skipping {pred_path.name}: unable to parse image id.")
            continue

        label_path = labels_dir / f'label_{img_num}.nii.gz'
        if not label_path.exists():
            fallback = labels_dir / f'label_{img_num}.nii'
            if fallback.exists():
                label_path = fallback
            else:
                print(f"Skipping {pred_path.name}: label not found at {label_path} or {fallback}.")
                continue

        pred = nib.load(str(pred_path)).get_fdata().astype(np.float32)
        label = nib.load(str(label_path)).get_fdata().astype(np.float32)

        dice, cldice, _ = calculate_metrics(pred, label, threshold=threshold, target_skeleton=None)
        rows.append((f'image_{img_num}', dice, cldice))
        print(f"image_{img_num}: Dice={dice:.4f}, clDice={cldice:.4f}")

    if not rows:
        print("No matching prediction/label pairs evaluated.")
        return

    dice_vals = [r[1] for r in rows]
    cldice_vals = [r[2] for r in rows]
    print("\n" + "=" * 60)
    print("PREDICTION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated: {len(rows)}")
    print(f"Dice Mean ± Std: {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
    print(f"Dice Min/Max: {np.min(dice_vals):.4f} / {np.max(dice_vals):.4f}")
    print(f"clDice Mean ± Std: {np.mean(cldice_vals):.4f} ± {np.std(cldice_vals):.4f}")
    print(f"clDice Min/Max: {np.min(cldice_vals):.4f} / {np.max(cldice_vals):.4f}")

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'dice', 'cldice'])
            writer.writerows(rows)
        print(f"Saved per-image metrics to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate saved ensemble predictions against labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--pred_dir', type=Path, default=Path('./results/predictions_ensemble'),
                        help='Directory containing pred_XXX.nii.gz files')
    parser.add_argument('--labels_dir', type=Path, default=Path('./data/labels'),
                        help='Directory containing label_XXX.nii.gz files')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing predictions')
    parser.add_argument('--csv_path', type=Path, default=None,
                        help='Optional path to save per-image metrics CSV')
    args = parser.parse_args()

    evaluate_predictions(args.pred_dir, args.labels_dir, args.threshold, args.csv_path)


if __name__ == '__main__':
    main()
