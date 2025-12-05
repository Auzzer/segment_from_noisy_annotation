"""Plot per-image line plots for model vs ensemble metrics."""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Reuse existing inference + metrics utilities
from model_script.predict.predict_ensemble import (
    calculate_metrics,
    load_ensemble_models,
    predict_with_ensemble,
)

def try_plot_from_csv(output_dir: Path, out_prefix=''):
    import csv
    csv_path = output_dir / 'per_image_metrics.csv'
    if not csv_path.exists():
        return False
    image_ids = []
    ensemble_dice, ensemble_cldice = [], []
    per_model_dice, per_model_cldice = None, None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False
        # determine number of models
        model_dice_cols = [i for i, h in enumerate(header) if h.endswith('_dice') and h.startswith('model_')]
        model_cldice_cols = [i for i, h in enumerate(header) if h.endswith('_cldice') and h.startswith('model_')]
        n_models = len(model_dice_cols)
        per_model_dice = [[] for _ in range(n_models)]
        per_model_cldice = [[] for _ in range(n_models)]
        for row in reader:
            image_ids.append(row[0])
            ensemble_dice.append(float(row[1]))
            ensemble_cldice.append(float(row[2]))
            for m_idx, col in enumerate(model_dice_cols):
                per_model_dice[m_idx].append(float(row[col]))
            for m_idx, col in enumerate(model_cldice_cols):
                per_model_cldice[m_idx].append(float(row[col]))

    # Plot dice
    make_line_plot(
        x_ids=image_ids,
        per_model_values=per_model_dice,
        ensemble_values=ensemble_dice,
        ylabel='Dice',
        title='Per-image Dice: Models vs Ensemble',
        out_path=output_dir / f'{out_prefix}dice_over_images_lines.png',
    )
    # Plot cldice
    make_line_plot(
        x_ids=image_ids,
        per_model_values=per_model_cldice,
        ensemble_values=ensemble_cldice,
        ylabel='clDice',
        title='Per-image clDice: Models vs Ensemble',
        out_path=output_dir / f'{out_prefix}cldice_over_images_lines.png',
    )
    print(f"Plotted from CSV: {csv_path}")
    return True


def make_line_plot(x_ids, per_model_values, ensemble_values, ylabel, title, out_path):
    x = np.arange(len(x_ids))
    fig, ax = plt.subplots(figsize=(max(8, len(x_ids) * 0.12), 4.5))

    # Plot each model as a faint line
    for model_idx, values in enumerate(per_model_values):
        ax.plot(x, values, lw=1.0, alpha=0.35, label=(f"Model {model_idx}" if model_idx < 10 else None))

    # Plot ensemble as a bold black line
    ax.plot(x, ensemble_values, lw=2.5, color='black', label='Ensemble')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)

    # Keep labels sparse if many images
    if len(x_ids) > 30:
        step = max(1, len(x_ids) // 30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([x_ids[i] for i in range(0, len(x_ids), step)], rotation=45, ha='right')
    else:
        ax.set_xticklabels(x_ids, rotation=45, ha='right')

    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(ncol=2, fontsize='small', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot per-image lines for model vs ensemble metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model/data parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./results/output_ensemble',
                        help='Directory containing trained ensemble models (with checkpoints/)')
    parser.add_argument('--num_models', type=int, default=10,
                        help='Number of models in the ensemble')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory (images/, labels/)')
    parser.add_argument('--output_dir', type=str, default='./results/predictions_ensemble',
                        help='Path to save line plots')
    parser.add_argument('--image_list', type=str, default=None,
                        help='Optional path to text file with image ids to process (image_XXX)')

    # Inference parameters
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Patch size for inference (D H W)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio between patches (0-1)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--patch_batch_size', type=int, default=1,
                        help='Number of patches to forward simultaneously during inference')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed-precision (autocast) inference on CUDA')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.amp and device.type != 'cuda':
        args.amp = False

    # If metrics CSV exists in output_dir, plot directly from it
    if try_plot_from_csv(Path(args.output_dir)):
        return

    # Otherwise load models and recompute
    models = load_ensemble_models(
        checkpoint_dir=Path(args.checkpoint_dir) / 'checkpoints',
        num_models=args.num_models,
        device=device,
    )
    if len(models) == 0:
        print('No models loaded. Exiting.')
        return

    # Prepare IO
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.data_dir) / 'images'
    labels_dir = Path(args.data_dir) / 'labels'

    # Collect image files, optionally filter by list
    image_files = sorted(list(images_dir.glob('*.nii.gz')))
    ids = []
    if args.image_list:
        # Normalize to canonical names image_XXX and filter
        from model_script.predict.predict_ensemble import load_image_id_list, normalize_image_id
        allowed = load_image_id_list(Path(args.image_list))
        filtered = []
        for p in image_files:
            if normalize_image_id(p.stem) in allowed:
                filtered.append(p)
                ids.append(normalize_image_id(p.stem))
        image_files = filtered
    else:
        ids = [p.stem.split('.')[0] for p in image_files]

    if not image_files:
        print('No images to process. Exiting.')
        return

    # Metric containers
    per_model_dice = [[] for _ in range(len(models))]
    per_model_cldice = [[] for _ in range(len(models))]
    ensemble_dice = []
    ensemble_cldice = []

    # Process images
    for img_path in image_files:
        img_num = img_path.stem.split('.')[0].split('_')[1]
        img = nib.load(str(img_path))
        image = img.get_fdata().astype(np.float32)

        # Predict
        pred, model_preds = predict_with_ensemble(
            models,
            image,
            device,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            patch_batch_size=args.patch_batch_size,
            use_amp=args.amp,
        )

        # Metrics
        label_path = labels_dir / f'label_{img_num}.nii.gz'
        if not label_path.exists():
            continue
        lbl = nib.load(str(label_path)).get_fdata().astype(np.float32)

        d, cd, skel = calculate_metrics(pred, lbl, threshold=args.threshold, target_skeleton=None)
        ensemble_dice.append(d)
        ensemble_cldice.append(cd)

        for m_idx, mp in enumerate(model_preds):
            md, mcd, _ = calculate_metrics(mp, lbl, threshold=args.threshold, target_skeleton=skel)
            per_model_dice[m_idx].append(md)
            per_model_cldice[m_idx].append(mcd)

    # Build x-axis IDs (ensure they match accumulated metrics length)
    n = len(ensemble_dice)
    if n == 0:
        print('No labeled images evaluated. Exiting.')
        return
    if len(ids) != n:
        # Reduce ids to those with labels encountered
        ids = ids[:n]

    # Plot Dice lines
    make_line_plot(
        x_ids=ids,
        per_model_values=per_model_dice,
        ensemble_values=ensemble_dice,
        ylabel='Dice',
        title='Per-image Dice: Models vs Ensemble',
        out_path=output_dir / 'dice_over_images_lines.png',
    )

    # Plot clDice lines
    make_line_plot(
        x_ids=ids,
        per_model_values=per_model_cldice,
        ensemble_values=ensemble_cldice,
        ylabel='clDice',
        title='Per-image clDice: Models vs Ensemble',
        out_path=output_dir / 'cldice_over_images_lines.png',
    )

    # Optional: save CSV with metrics
    try:
        import csv
        csv_path = output_dir / 'per_image_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['image_id', 'ensemble_dice', 'ensemble_cldice'] + \
                     [f'model_{i}_dice' for i in range(len(models))] + \
                     [f'model_{i}_cldice' for i in range(len(models))]
            writer.writerow(header)
            for i in range(n):
                row = [ids[i], ensemble_dice[i], ensemble_cldice[i]]
                row += [per_model_dice[m][i] for m in range(len(models))]
                row += [per_model_cldice[m][i] for m in range(len(models))]
                writer.writerow(row)
        print(f"Saved metrics CSV to: {csv_path}")
    except Exception as e:
        print(f"Warning: failed to write CSV ({e})")


if __name__ == '__main__':
    main()
