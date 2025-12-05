import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_erosion, generate_binary_structure
from contextlib import nullcontext
from torch.cuda.amp import autocast
import csv

try:
    from monai.metrics import compute_meandice as monai_compute_dice
except ImportError:
    from monai.metrics import compute_dice as monai_compute_dice
    
EPS = 1e-8

from utils.unet_model import UNet3D


def normalize_image_id(name: str) -> str:
    base = name.split('.')[0]
    if base.startswith('image_'):
        return base
    if base.startswith('label_'):
        suffix = base.split('_', 1)[1]
        return f'image_{suffix}'
    if base.isdigit():
        return f'image_{int(base):03d}'
    return base


def load_image_id_list(list_path: Path):
    ids = set()
    with open(list_path, 'r') as f:
        for line in f:
            entry = line.strip()
            if not entry:
                continue
            ids.add(normalize_image_id(entry))
    if not ids:
        raise ValueError(f"No valid entries found in {list_path}")
    return ids


def morphological_skeleton(binary_image):
    """Compute morphological skeleton of a binary image using iterative erosion."""
    binary_image = binary_image.astype(bool)
    structure = generate_binary_structure(binary_image.ndim, 1)
    skeleton = np.zeros_like(binary_image, dtype=bool)
    eroded = binary_image.copy()
    
    while eroded.any():
        opened = binary_opening(eroded, structure=structure)
        skeleton_layer = eroded & (~opened)
        skeleton |= skeleton_layer
        eroded = binary_erosion(eroded, structure=structure)
    
    return skeleton


def compute_cldice_score(pred_binary, target_binary, target_skeleton=None):
    """
    Compute centerline Dice (clDice) between binary prediction and target.
    
    Args:
        pred_binary: Binary numpy array of the prediction (bool or 0/1).
        target_binary: Binary numpy array of the target (bool or 0/1).
        target_skeleton: Optional precomputed skeleton of the target.
    
    Returns:
        Tuple of (clDice score, target skeleton for potential reuse).
    """
    pred_binary = pred_binary.astype(bool)
    target_binary = target_binary.astype(bool)
    
    pred_skeleton = morphological_skeleton(pred_binary)
    target_skeleton = (
        target_skeleton if target_skeleton is not None else morphological_skeleton(target_binary)
    )
    
    pred_skel_sum = pred_skeleton.sum()
    target_skel_sum = target_skeleton.sum()
    
    if pred_skel_sum == 0 and target_skel_sum == 0:
        return 1.0, target_skeleton
    if pred_skel_sum == 0 or target_skel_sum == 0:
        return 0.0, target_skeleton
    
    topology_precision = (
        np.logical_and(pred_skeleton, target_binary).sum() / (pred_skel_sum + EPS)
    )
    topology_sensitivity = (
        np.logical_and(target_skeleton, pred_binary).sum() / (target_skel_sum + EPS)
    )
    
    if topology_precision + topology_sensitivity == 0:
        return 0.0, target_skeleton
    
    cldice = (2 * topology_precision * topology_sensitivity) / (
        topology_precision + topology_sensitivity + EPS
    )
    
    return float(cldice), target_skeleton


def load_ensemble_models(checkpoint_dir, num_models, device):
    """Load all models in the ensemble"""
    models = []
    
    print(f"Loading {num_models} models from {checkpoint_dir}")
    for model_id in range(num_models):
        model_path = Path(checkpoint_dir) / f'model_{model_id}' / 'best.pth'
        
        if not model_path.exists():
            print(f"Warning: Model {model_id} checkpoint not found at {model_path}")
            continue
        
        # Create model
        model = UNet3D(n_channels=1, n_classes=1, trilinear=True)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        models.append(model)
        val_dice = checkpoint.get('val_dice')
        if isinstance(val_dice, (float, int)):
            print(f"  Loaded model {model_id} (Dice: {val_dice:.4f})")
        else:
            print(f"  Loaded model {model_id} (Dice: N/A)")
    
    print(f"Successfully loaded {len(models)} models")
    return models


def fuse_predictions(model_predictions, method="mean", threshold=0.5):
    """Fuse per-model predictions into a single volume."""
    if method == "mean":
        return np.mean(model_predictions, axis=0)

    if method == "staple":
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError(
                "SimpleITK is required for STAPLE fusion. Install with `pip install SimpleITK`."
            ) from exc

        # STAPLE expects binary segmentations stacked; threshold each model.
        sitk_labels = []
        for pred in model_predictions:
            binary = (pred > threshold).astype(np.uint8)
            sitk_img = sitk.GetImageFromArray(binary)
            sitk_labels.append(sitk_img)

        staple_filter = sitk.STAPLEImageFilter()
        fused_sitk = staple_filter.Execute(sitk_labels)
        fused = sitk.GetArrayFromImage(fused_sitk).astype(np.float32)
        return fused

    raise ValueError(f"Unknown fusion method '{method}' (expected 'mean' or 'staple').")


def predict_with_ensemble(
    models,
    image,
    device,
    patch_size=(64, 64, 64),
    overlap=0.5,
    patch_batch_size=1,
    use_amp=False,
    fusion_method="mean",
    fusion_threshold=0.5,
):
    """
    Make prediction using ensemble of models with sliding window approach
    
    Args:
        models: List of trained models
        image: Input image (D, H, W)
        device: Device to run inference on
        patch_size: Size of patches for inference
        overlap: Overlap ratio between patches (0-1)
        patch_batch_size: Number of patches to process simultaneously on the GPU
        use_amp: Enable mixed-precision autocast when running on CUDA
        fusion_method: How to combine per-model predictions ('mean' or 'staple')
        fusion_threshold: Threshold used to binarize per-model outputs for STAPLE
    
    Returns:
        Tuple of (
            averaged prediction from all models,
            list of per-model predictions in the same order as `models`
        )
    """
    d, h, w = image.shape
    pd, ph, pw = patch_size
    
    # Calculate stride based on overlap
    stride_d = int(pd * (1 - overlap))
    stride_h = int(ph * (1 - overlap))
    stride_w = int(pw * (1 - overlap))
    
    # Ensure strides are at least 1
    stride_d = max(1, stride_d)
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)
    
    # Initialize output arrays for averaging
    count_map = np.zeros((d, h, w), dtype=np.float32)
    model_prediction_sums = [
        np.zeros((d, h, w), dtype=np.float32) for _ in range(len(models))
    ]
    
    # Normalize image
    image_normalized = (image - image.mean()) / (image.std() + 1e-8)
    
    # Generate sliding window coordinates
    d_starts = list(range(0, max(1, d - pd + 1), stride_d))
    h_starts = list(range(0, max(1, h - ph + 1), stride_h))
    w_starts = list(range(0, max(1, w - pw + 1), stride_w))
    
    # Make sure we cover the entire image
    if d_starts[-1] + pd < d:
        d_starts.append(d - pd)
    if h_starts[-1] + ph < h:
        h_starts.append(h - ph)
    if w_starts[-1] + pw < w:
        w_starts.append(w - pw)
    
    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    print(f"Processing {total_patches} patches with {len(models)} models...")
    
    patch_batch_size = max(1, int(patch_batch_size))
    autocast_ctx = autocast if (use_amp and device.type == 'cuda') else nullcontext
    
    patch_tensors = []
    patch_coords = []
    
    with torch.no_grad():
        pbar = tqdm(total=total_patches, desc="Inference")
        
        def flush_batch():
            if not patch_tensors:
                return
            
            batch_tensor = torch.cat(patch_tensors, dim=0).to(device, non_blocking=True)
            
            with autocast_ctx():
                for model_idx, model in enumerate(models):
                    preds = model(batch_tensor)
                    preds = torch.sigmoid(preds)
                    preds_np = preds.detach().cpu().numpy()
                    
                    for i, (d_start_i, h_start_i, w_start_i) in enumerate(patch_coords):
                        model_prediction_sums[model_idx][
                            d_start_i:d_start_i+pd,
                            h_start_i:h_start_i+ph,
                            w_start_i:w_start_i+pw
                        ] += preds_np[i, 0]
            
            processed = len(patch_coords)
            patch_tensors.clear()
            patch_coords.clear()
            pbar.update(processed)
        
        for d_start in d_starts:
            for h_start in h_starts:
                for w_start in w_starts:
                    # Extract patch
                    patch = image_normalized[
                        d_start:d_start+pd,
                        h_start:h_start+ph,
                        w_start:w_start+pw
                    ]
                    
                    # Add batch and channel dimensions
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                    patch_tensors.append(patch_tensor)
                    patch_coords.append((d_start, h_start, w_start))
                    
                    count_map[
                        d_start:d_start+pd,
                        h_start:h_start+ph,
                        w_start:w_start+pw
                    ] += 1
                    
                    if len(patch_tensors) >= patch_batch_size:
                        flush_batch()
        
        # Flush any remaining patches
        flush_batch()
        
        pbar.close()
    
    # Average overlapping predictions per model
    model_predictions = [
        model_sum / (count_map + 1e-8) for model_sum in model_prediction_sums
    ]
    
    # Ensemble prediction using requested fusion strategy
    prediction = fuse_predictions(
        model_predictions, method=fusion_method, threshold=fusion_threshold
    )
    
    return prediction, model_predictions


def calculate_metrics(pred, target, threshold=0.5, target_skeleton=None):
    """Calculate Dice and clDice scores using MONAI Dice and custom clDice."""
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    
    pred_binary = (pred_tensor > threshold).float()
    target_binary = (target_tensor > 0.5).float()
    
    dice_tensor = monai_compute_dice(
        pred_binary, target_binary, include_background=True
    )
    dice = torch.nan_to_num(dice_tensor, nan=1.0).mean().item()
    
    pred_binary_np = pred_binary.cpu().numpy().astype(bool)[0, 0]
    target_binary_np = target_binary.cpu().numpy().astype(bool)[0, 0]
    
    cldice, target_skeleton = compute_cldice_score(
        pred_binary_np, target_binary_np, target_skeleton=target_skeleton
    )
    
    return float(dice), float(cldice), target_skeleton


def predict_on_dataset(args):
    """Run prediction on a dataset"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.amp and device.type != 'cuda':
        print("Warning: AMP requested but CUDA not available; disabling AMP.")
        args.amp = False
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load ensemble models
    models = load_ensemble_models(
        checkpoint_dir=Path(args.checkpoint_dir) / 'checkpoints',
        num_models=args.num_models,
        device=device
    )
    
    if len(models) == 0:
        print("Error: No models loaded!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to process
    images_dir = Path(args.data_dir) / 'images'
    labels_dir = Path(args.data_dir) / 'labels'
    
    image_files = sorted(list(images_dir.glob('*.nii.gz')))
    
    if args.image_list:
        allowed_ids = load_image_id_list(Path(args.image_list))
        filtered = []
        for img in image_files:
            normalized = normalize_image_id(img.stem)
            if normalized in allowed_ids:
                filtered.append(img)
        missing = allowed_ids - {normalize_image_id(img.stem) for img in filtered}
        if missing:
            print("Warning: the following ids were requested but not found:")
            for miss in sorted(missing):
                print(f"  - {miss}")
        image_files = filtered
        print(f"\nFiltered to {len(image_files)} images from list: {args.image_list}")
    else:
        print(f"\nFound {len(image_files)} images to process")
    
    # Prepare metrics containers and CSV writer
    ensemble_metrics = {"dice": [], "cldice": []}
    per_model_metrics = [{"dice": [], "cldice": []} for _ in models]
    csv_path = output_dir / 'per_image_metrics.csv'
    
    # Track processed images for resume functionality
    processed_images = set()
    if args.resume and csv_path.exists():
        print(f"Resume mode enabled. Loading existing results from {csv_path}")
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            for row in reader:
                if row:
                    processed_images.add(row[0])  # image_id is first column
        print(f"Found {len(processed_images)} already processed images. Skipping them...")
        print()
    
    # Determine write mode based on resume flag
    csv_mode = 'a' if (args.resume and csv_path.exists()) else 'w'
    write_header = not (args.resume and csv_path.exists())
    
    with open(csv_path, csv_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            header = (
                ['image_id', 'ensemble_dice', 'ensemble_cldice'] +
                [f'model_{i}_dice' for i in range(len(models))] +
                [f'model_{i}_cldice' for i in range(len(models))]
            )
            writer.writerow(header)
        
        for img_file in tqdm(image_files, desc="Processing images"):
            # Extract image number
            img_num = img_file.stem.split('.')[0].split('_')[1]
            img_id = f"image_{img_num}"
            
            # Skip if already processed (resume mode)
            if img_id in processed_images:
                continue
            
            # Load image
            img_nifti = nib.load(str(img_file))
            image = img_nifti.get_fdata().astype(np.float32)
            
            # Make prediction
            prediction, model_predictions = predict_with_ensemble(
                models,
                image,
                device,
                patch_size=tuple(args.patch_size),
                overlap=args.overlap,
                patch_batch_size=args.patch_batch_size,
                use_amp=args.amp,
                fusion_method=args.fusion_method,
                fusion_threshold=args.threshold,
            )
            
            # Save prediction (ensemble)
            pred_nifti = nib.Nifti1Image(prediction, img_nifti.affine, img_nifti.header)
            pred_path = output_dir / f'pred_{img_num}.nii.gz'
            nib.save(pred_nifti, str(pred_path))
            
            # Optionally save per-model predictions
            if args.save_per_model_preds:
                for m_idx, m_pred in enumerate(model_predictions):
                    m_dir = output_dir / 'per_model' / f'model_{m_idx}'
                    m_dir.mkdir(parents=True, exist_ok=True)
                    m_path = m_dir / f'pred_{img_num}.nii.gz'
                    m_nifti = nib.Nifti1Image(m_pred, img_nifti.affine, img_nifti.header)
                    nib.save(m_nifti, str(m_path))
            
            # Calculate Dice score if label exists
            label_file = labels_dir / f'label_{img_num}.nii.gz'
            if label_file.exists():
                label_nifti = nib.load(str(label_file))
                label = label_nifti.get_fdata().astype(np.float32)
                
                label_skeleton = None
                ensemble_dice, ensemble_cldice, label_skeleton = calculate_metrics(
                    prediction, label, threshold=args.threshold, target_skeleton=None
                )
                ensemble_metrics["dice"].append(ensemble_dice)
                ensemble_metrics["cldice"].append(ensemble_cldice)
                print(
                    f"  Image {img_num}: Ensemble Dice = {ensemble_dice:.4f}, "
                    f"clDice = {ensemble_cldice:.4f}"
                )
                
                row = [img_id, ensemble_dice, ensemble_cldice]
                model_dice_vals = []
                model_cldice_vals = []
                for model_idx, model_pred in enumerate(model_predictions):
                    model_dice, model_cldice, _ = calculate_metrics(
                        model_pred, label,
                        threshold=args.threshold,
                        target_skeleton=label_skeleton
                    )
                    per_model_metrics[model_idx]["dice"].append(model_dice)
                    per_model_metrics[model_idx]["cldice"].append(model_cldice)
                    model_dice_vals.append(model_dice)
                    model_cldice_vals.append(model_cldice)
                    print(
                        f"    Model {model_idx}: Dice = {model_dice:.4f}, "
                        f"clDice = {model_cldice:.4f}"
                    )
                row.extend(model_dice_vals)
                row.extend(model_cldice_vals)
                writer.writerow(row)
                csvfile.flush()
    
    # Print summary
    if ensemble_metrics["dice"]:
        print("\n" + "="*80)
        print("PREDICTION SUMMARY")
        print("="*80)
        
        dice_mean = np.mean(ensemble_metrics["dice"])
        dice_std = np.std(ensemble_metrics["dice"])
        print(f"Ensemble Average Dice Score: {dice_mean:.4f} ± {dice_std:.4f}")
        print(f"Ensemble Dice Min/Max: {np.min(ensemble_metrics['dice']):.4f} / {np.max(ensemble_metrics['dice']):.4f}")
        
        if ensemble_metrics["cldice"]:
            cldice_mean = np.mean(ensemble_metrics["cldice"])
            cldice_std = np.std(ensemble_metrics["cldice"])
            print(f"Ensemble Average clDice Score: {cldice_mean:.4f} ± {cldice_std:.4f}")
            print(
                f"Ensemble clDice Min/Max: "
                f"{np.min(ensemble_metrics['cldice']):.4f} / {np.max(ensemble_metrics['cldice']):.4f}"
            )
        
        print("\nPer-model averages:")
        for model_idx, metrics in enumerate(per_model_metrics):
            if metrics["dice"]:
                model_dice_mean = np.mean(metrics["dice"])
                model_dice_std = np.std(metrics["dice"])
                msg = f"  Model {model_idx}: Dice {model_dice_mean:.4f} ± {model_dice_std:.4f}"
                if metrics["cldice"]:
                    model_cldice_mean = np.mean(metrics["cldice"])
                    model_cldice_std = np.std(metrics["cldice"])
                    msg += f", clDice {model_cldice_mean:.4f} ± {model_cldice_std:.4f}"
                print(msg)
            else:
                print(f"  Model {model_idx}: No labels available for evaluation")

        # Generate bar plot comparing models and ensemble (Dice)
        labels = [f"Model {idx}" for idx in range(len(per_model_metrics))] + ["Ensemble"]
        dice_values = [
            np.mean(metrics["dice"]) if metrics["dice"] else 0.0
            for metrics in per_model_metrics
        ] + [dice_mean]
        dice_errors = [
            np.std(metrics["dice"]) if len(metrics["dice"]) > 1 else 0.0
            for metrics in per_model_metrics
        ] + ([dice_std] if len(ensemble_metrics["dice"]) > 1 else [0.0])
        
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
        ax.bar(x, dice_values, yerr=dice_errors, capsize=5, color='steelblue')
        ax.set_ylabel('Dice Score')
        ax.set_title('Model vs Ensemble Dice Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        
        dice_plot_path = output_dir / 'dice_scores_bar.png'
        fig.savefig(dice_plot_path, dpi=300)
        plt.close(fig)
        print(f"\nDice bar plot saved to: {dice_plot_path}")
        
        if ensemble_metrics["cldice"]:
            cldice_mean = np.mean(ensemble_metrics["cldice"])
            cldice_std = np.std(ensemble_metrics["cldice"])
            cldice_values = [
                np.mean(metrics["cldice"]) if metrics["cldice"] else 0.0
                for metrics in per_model_metrics
            ] + [cldice_mean]
            cldice_errors = [
                np.std(metrics["cldice"]) if len(metrics["cldice"]) > 1 else 0.0
                for metrics in per_model_metrics
            ] + ([cldice_std] if len(ensemble_metrics["cldice"]) > 1 else [0.0])
            
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
            ax.bar(x, cldice_values, yerr=cldice_errors, capsize=5, color='seagreen')
            ax.set_ylabel('clDice Score')
            ax.set_title('Model vs Ensemble clDice Scores')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            fig.tight_layout()
            
            cldice_plot_path = output_dir / 'cldice_scores_bar.png'
            fig.savefig(cldice_plot_path, dpi=300)
            plt.close(fig)
            print(f"clDice bar plot saved to: {cldice_plot_path}")
    
    print(f"\nPredictions saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict using ensemble of 3D U-Nets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing trained ensemble models')
    parser.add_argument('--num_models', type=int, default=10,
                        help='Number of models in the ensemble')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./results/predictions_ensemble',
                        help='Path to save predictions')
    parser.add_argument('--image_list', type=str, default=None,
                        help='Optional text file with image ids to process (e.g., from generated test list)')
    
    # Inference parameters
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Patch size for inference (D H W)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio between patches (0-1)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--fusion_method', type=str, default='mean', choices=['mean', 'staple'],
                        help="How to fuse model predictions: 'mean' (default) or 'staple' (requires SimpleITK)")
    parser.add_argument('--patch_batch_size', type=int, default=1,
                        help='Number of patches to forward simultaneously during inference')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed-precision (autocast) inference on CUDA')
    parser.add_argument('--save_per_model_preds', action='store_true',
                        help='Save each model\'s prediction as NIfTI under output_dir/per_model/model_i')
    parser.add_argument('--resume', action='store_true',
                        help='Resume prediction from where it was interrupted (skip already processed images)')
    
    args = parser.parse_args()
    
    predict_on_dataset(args)


if __name__ == '__main__':
    main()
