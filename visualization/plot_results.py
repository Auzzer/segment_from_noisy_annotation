"""
Plot training and ensemble results.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def plot_training_history(checkpoint_path: Path):
    """
    Plot training history from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Training may still be in progress.")
        return

    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print("\nFinal Metrics:")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Train Dice: {checkpoint['train_dice']:.4f}")
    print(f"  Val Loss:   {checkpoint['val_loss']:.4f}")
    print(f"  Val Dice:   {checkpoint['val_dice']:.4f}")
    print("=" * 60 + "\n")
    print("Note: For full training curves, use TensorBoard:")
    print("  ./noisy_seg_env/bin/tensorboard --logdir ./results/single_model/logs")
    print("  Then open http://localhost:6006 in your browser")


def _bar_plot(values, errors, labels, title, ylabel, out_path: Path):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
    ax.bar(x, values, yerr=errors, capsize=5, color='steelblue')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ensemble_bars_from_csv(metrics_csv: Path, out_dir: Path):
    if not metrics_csv.exists():
        print(f"Metrics CSV not found: {metrics_csv}")
        return False

    with metrics_csv.open('r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("Empty metrics CSV.")
            return False

        model_dice_cols = [i for i, h in enumerate(header) if h.startswith('model_') and h.endswith('_dice')]
        model_cldice_cols = [i for i, h in enumerate(header) if h.startswith('model_') and h.endswith('_cldice')]
        n_models = len(model_dice_cols)

        ens_dice, ens_cldice = [], []
        per_model_dice = [[] for _ in range(n_models)]
        per_model_cldice = [[] for _ in range(n_models)]

        for row in reader:
            ens_dice.append(float(row[1]))
            ens_cldice.append(float(row[2]))
            for m_idx, col in enumerate(model_dice_cols):
                per_model_dice[m_idx].append(float(row[col]))
            for m_idx, col in enumerate(model_cldice_cols):
                per_model_cldice[m_idx].append(float(row[col]))

    dice_means = [np.mean(x) if x else 0.0 for x in per_model_dice]
    dice_stds = [np.std(x) if len(x) > 1 else 0.0 for x in per_model_dice]
    ens_dice_mean = float(np.mean(ens_dice)) if ens_dice else 0.0
    ens_dice_std = float(np.std(ens_dice)) if len(ens_dice) > 1 else 0.0

    cldice_means = [np.mean(x) if x else 0.0 for x in per_model_cldice]
    cldice_stds = [np.std(x) if len(x) > 1 else 0.0 for x in per_model_cldice]
    ens_cldice_mean = float(np.mean(ens_cldice)) if ens_cldice else 0.0
    ens_cldice_std = float(np.std(ens_cldice)) if len(ens_cldice) > 1 else 0.0

    labels = [f"Model {i}" for i in range(n_models)] + ["Ensemble"]
    dice_values = dice_means + [ens_dice_mean]
    dice_errors = dice_stds + [ens_dice_std]
    cldice_values = cldice_means + [ens_cldice_mean]
    cldice_errors = cldice_stds + [ens_cldice_std]

    out_dir.mkdir(parents=True, exist_ok=True)
    _bar_plot(dice_values, dice_errors, labels, 'Model vs Ensemble Dice Scores', 'Dice Score', out_dir / 'dice_scores_bar.png')
    _bar_plot(cldice_values, cldice_errors, labels, 'Model vs Ensemble clDice Scores', 'clDice Score', out_dir / 'cldice_scores_bar.png')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Plot training or ensemble results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=Path, default=None, 
                        help='Path to a training checkpoint to summarize')
    parser.add_argument('--metrics_csv', type=Path, default=Path('./results/output_ensemble/predictions/per_image_metrics.csv'),
                        help='Path to per-image metrics CSV for ensemble plots')
    parser.add_argument('--out_dir', type=Path, default=Path('./results/output_ensemble/predictions'),
                        help='Directory to save generated figures for ensemble plots')
    args = parser.parse_args()

    did_anything = False
    if args.checkpoint is not None:
        plot_training_history(args.checkpoint)
        did_anything = True

    if plot_ensemble_bars_from_csv(args.metrics_csv, args.out_dir):
        did_anything = True

    if not did_anything:
        best_path = Path('./results/output_ensemble/checkpoints/best.pth')
        latest_path = Path('./results/output_ensemble/checkpoints/latest.pth')
        if best_path.exists():
            plot_training_history(best_path)
        elif latest_path.exists():
            plot_training_history(latest_path)
        else:
            print('Nothing to plot: provide --checkpoint or generate metrics CSV first.')


if __name__ == '__main__':
    main()

