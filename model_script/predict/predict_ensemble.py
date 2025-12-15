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

"""
Example (iso inputs):
  python -m model_script.predict.predict_ensemble \
    --checkpoint_template "/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/Dataset001_nnunet/seed_{seed}/Dataset001_nnunet/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/checkpoint_best.pth" \
    --fold 0 --num_models 10 \
    --image_source iso \
    --images_dir ./data_preco/images_iso \
    --labels_dir ./data_preco/labels_iso \
    --mean_output_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/mean \
    --staple_output_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/staple

Example (orig inputs):
  python -m model_script.predict.predict_ensemble \
    --checkpoint_template "/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/Dataset001_nnunet/seed_{seed}/Dataset001_nnunet/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/checkpoint_best.pth" \
    --fold 0 --num_models 10 \
    --image_source orig --orig_images_dir ./data/images --orig_labels_dir ./data/label\
    --mean_output_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/mean \
    --staple_output_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/staple
"""

try:
    from monai.metrics import compute_meandice as monai_compute_dice
except ImportError:
    from monai.metrics import compute_dice as monai_compute_dice
    
EPS = 1e-8

from utils.unet_model import UNet3D

# Helpers for orientation (SimpleITK returns arrays as (z, y, x); convert to (x, y, z))
def sitk_to_nib_order(arr: np.ndarray) -> np.ndarray:
    return np.transpose(arr, (2, 1, 0))


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


def log_stats(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    print(
        f"{name}: shape {arr.shape}, min {arr.min():.4f}, max {arr.max():.4f}, mean {arr.mean():.4f}"
    )


def load_ensemble_models(checkpoint_template: str, num_models: int, fold: int, device):
    """Load all models in the ensemble using a seed-indexed checkpoint template."""
    models = []
    
    print(f"Loading {num_models} models from template: {checkpoint_template} (fold={fold})")
    for seed in range(1, num_models + 1):
        try:
            model_path = Path(checkpoint_template.format(seed=seed, fold=fold))
        except KeyError as exc:
            raise ValueError(
                "checkpoint_template must contain '{seed}' and '{fold}' placeholders for numbering."
            ) from exc
        
        if not model_path.exists():
            print(f"Warning: Model seed {seed} checkpoint not found at {model_path}")
            continue
        
        # Torch 2.6 defaults to weights_only=True, which breaks older checkpoints; force False with fallback
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        else:
            state = checkpoint
        # Strip DistributedDataParallel prefixes if present
        if isinstance(state, dict):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        else:
            raise ValueError(f"Unexpected checkpoint structure for {model_path}")

        # Infer number of classes from the head shape (supports legacy 1-ch and new 2-ch heads)
        head_w = state.get('outc.conv.weight')
        n_classes = head_w.shape[0] if head_w is not None else 1
        if n_classes not in (1, 2):
            print(f"  Warning: unexpected head channels={n_classes}; defaulting to 1")
            n_classes = 1

        model = UNet3D(n_channels=1, n_classes=n_classes, trilinear=True)
        # If checkpoint is legacy 1-channel and model is 2-channel, map weights
        model_state = model.state_dict()
        filtered_state = {}
        if n_classes == 2 and head_w is not None and head_w.shape[0] == 1:
            new_w = model_state['outc.conv.weight'].clone()
            new_b = model_state['outc.conv.bias'].clone()
            new_w.zero_()
            new_b.zero_()
            new_w[1] = head_w[0]
            new_b[1] = state['outc.conv.bias'][0]
            filtered_state['outc.conv.weight'] = new_w
            filtered_state['outc.conv.bias'] = new_b
            print(f"  Seed {seed}: expanded 1->2 channel head during load.")

        for k, v in state.items():
            if n_classes == 2 and k in ('outc.conv.weight', 'outc.conv.bias') and head_w.shape[0] == 1:
                continue
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v

        model.load_state_dict(filtered_state, strict=False)
        model = model.to(device)
        model.eval()
        
        models.append(model)
        val_dice = checkpoint.get('val_dice') if isinstance(checkpoint, dict) else None
        if isinstance(val_dice, (float, int)):
            print(f"  Loaded seed {seed} (Dice: {val_dice:.4f})")
        else:
            print(f"  Loaded seed {seed} (Dice: N/A)")
    
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


def load_lung_mask(mask_dir: Path, img_num: str, expected_shape, image_path: Path) -> np.ndarray:
    """Load and binarize a lung mask for the given image number, resampling if needed."""
    candidates = [
        mask_dir / f"lung_mask_{img_num}.nii.gz",
        mask_dir / f"lung_mask_image_{img_num}.nii.gz",
    ]
    mask_path = next((p for p in candidates if p.exists()), None)
    if mask_path is None:
        raise FileNotFoundError(f"Lung mask not found for image {img_num} in {mask_dir}")

    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("SimpleITK is required to load/resample lung masks.") from exc

    mask_img_sitk = sitk.ReadImage(str(mask_path))
    mask_np = sitk_to_nib_order(sitk.GetArrayFromImage(mask_img_sitk))
    mask_np = (mask_np > 0).astype(np.float32)
    if mask_np.shape == expected_shape:
        return mask_np

    # Resample mask to match image geometry
    target_img_sitk = sitk.ReadImage(str(image_path))
    resampled = sitk.Resample(
        mask_img_sitk,
        target_img_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        mask_img_sitk.GetPixelID(),
    )
    mask_resampled = sitk_to_nib_order(sitk.GetArrayFromImage(resampled))
    mask_resampled = (mask_resampled > 0).astype(np.float32)
    if mask_resampled.shape != expected_shape:
        raise ValueError(
            f"Lung mask shape {mask_resampled.shape} still mismatches image shape {expected_shape} for {img_num} after resampling."
        )
    return mask_resampled


def load_label_resampled(label_path: Path, image_path: Path, expected_shape) -> np.ndarray:
    """Load label and resample to match the given image geometry if needed."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("SimpleITK is required to resample labels.") from exc

    label_img = sitk.ReadImage(str(label_path))
    target_img = sitk.ReadImage(str(image_path))

    if label_img.GetSize() != target_img.GetSize():
        resampled = sitk.Resample(
            label_img,
            target_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            label_img.GetPixelID(),
        )
    else:
        resampled = label_img

    label_np = sitk_to_nib_order(sitk.GetArrayFromImage(resampled)).astype(np.float32)
    if label_np.shape != expected_shape:
        raise ValueError(
            f"Label shape {label_np.shape} still mismatches image shape {expected_shape} after resampling."
        )
    return label_np


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
                    if preds.shape[1] == 1:
                        preds = torch.sigmoid(preds)
                        preds_np = preds.detach().cpu().numpy()[:, 0]
                    else:
                        preds = torch.softmax(preds, dim=1)
                        preds_np = preds.detach().cpu().numpy()[:, 1]
                    
                    for i, (d_start_i, h_start_i, w_start_i) in enumerate(patch_coords):
                        model_prediction_sums[model_idx][
                            d_start_i:d_start_i+pd,
                            h_start_i:h_start_i+ph,
                            w_start_i:w_start_i+pw
                        ] += preds_np[i]
            
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


def predict_single_model(
    model,
    image: np.ndarray,
    device,
    patch_size=(64, 64, 64),
    overlap=0.5,
    patch_batch_size=1,
    use_amp=False,
):
    """
    Sliding-window prediction for a single model, returning a full-volume probability map.
    """
    d, h, w = image.shape
    pd, ph, pw = patch_size

    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))

    count_map = np.zeros((d, h, w), dtype=np.float32)
    pred_sum = np.zeros((d, h, w), dtype=np.float32)

    image_normalized = (image - image.mean()) / (image.std() + 1e-8)

    d_starts = list(range(0, max(1, d - pd + 1), stride_d))
    h_starts = list(range(0, max(1, h - ph + 1), stride_h))
    w_starts = list(range(0, max(1, w - pw + 1), stride_w))
    if d_starts[-1] + pd < d:
        d_starts.append(d - pd)
    if h_starts[-1] + ph < h:
        h_starts.append(h - ph)
    if w_starts[-1] + pw < w:
        w_starts.append(w - pw)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    patch_batch_size = max(1, int(patch_batch_size))
    autocast_ctx = autocast if (use_amp and device.type == 'cuda') else nullcontext

    patch_tensors = []
    patch_coords = []

    with torch.no_grad():
        pbar = tqdm(total=total_patches, desc="Inference (per model)")

        def flush_batch():
            if not patch_tensors:
                return
            batch_tensor = torch.cat(patch_tensors, dim=0).to(device, non_blocking=True)
            with autocast_ctx():
                preds = model(batch_tensor)
                if preds.shape[1] == 1:
                    preds = torch.sigmoid(preds)
                    preds_np = preds.detach().cpu().numpy()[:, 0]
                else:
                    preds = torch.softmax(preds, dim=1)
                    preds_np = preds.detach().cpu().numpy()[:, 1]
            for i, (d_start_i, h_start_i, w_start_i) in enumerate(patch_coords):
                pred_sum[
                    d_start_i:d_start_i+pd,
                    h_start_i:h_start_i+ph,
                    w_start_i:w_start_i+pw
                ] += preds_np[i]
                count_map[
                    d_start_i:d_start_i+pd,
                    h_start_i:h_start_i+ph,
                    w_start_i:w_start_i+pw
                ] += 1
            processed = len(patch_coords)
            patch_tensors.clear()
            patch_coords.clear()
            pbar.update(processed)

        for d_start in d_starts:
            for h_start in h_starts:
                for w_start in w_starts:
                    patch = image_normalized[
                        d_start:d_start+pd,
                        h_start:h_start+ph,
                        w_start:w_start+pw
                    ]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                    patch_tensors.append(patch_tensor)
                    patch_coords.append((d_start, h_start, w_start))
                    if len(patch_tensors) >= patch_batch_size:
                        flush_batch()
        flush_batch()
        pbar.close()

    prediction = pred_sum / (count_map + 1e-8)
    return prediction


def calculate_metrics(pred, target, threshold=0.5, target_skeleton=None, compute_cldice=True):
    """Calculate Dice and optional clDice scores using MONAI Dice and custom clDice."""
    log_stats("Metric input pred", pred)
    log_stats("Metric input target", target)
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
    
    pred_binary = (pred_tensor > threshold).float()
    target_binary = (target_tensor > 0.5).float()
    
    dice_tensor = monai_compute_dice(
        pred_binary, target_binary, include_background=True
    )
    dice = torch.nan_to_num(dice_tensor, nan=1.0).mean().item()
    
    cldice = None
    if compute_cldice:
        pred_binary_np = pred_binary.cpu().numpy().astype(bool)[0, 0]
        target_binary_np = target_binary.cpu().numpy().astype(bool)[0, 0]
        
        cldice, target_skeleton = compute_cldice_score(
            pred_binary_np, target_binary_np, target_skeleton=target_skeleton
        )
        cldice = float(cldice)
    
    return float(dice), cldice, target_skeleton


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
        checkpoint_template=args.checkpoint_template,
        fold=args.fold,
        num_models=args.num_models,
        device=device
    )
    
    if len(models) == 0:
        print("Error: No models loaded!")
        return
    
    compute_metrics = not args.skip_metrics
    if args.skip_metrics and args.resume:
        print("Resume flag ignored because metrics are skipped; all images will be processed.")
    
    def prepare_for_saving(volume: np.ndarray) -> np.ndarray:
        """Optionally binarize prediction before writing to disk."""
        if args.binarize_saved_outputs:
            return (volume > args.threshold).astype(np.uint8)
        return volume
    
    # Create output directories
    mean_output_dir = Path(args.mean_output_dir)
    staple_output_dir = Path(args.staple_output_dir)
    mean_output_dir.mkdir(parents=True, exist_ok=True)
    staple_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = mean_output_dir  # used for metrics/csv/plots
    lung_mask_dir = Path(args.lung_mask_dir)
    per_model_output_root = (
        Path(args.per_model_output_root)
        if args.per_model_output_root is not None
        else mean_output_dir / 'per_model'
    )
    per_model_output_root.mkdir(parents=True, exist_ok=True)
    print(f"Per-model predictions will be saved under: {per_model_output_root}")
    
    # Get list of images to process
    images_dir = Path(args.images_dir if args.image_source == 'iso' else args.orig_images_dir)
    labels_dir = Path(args.labels_dir if args.image_source == 'iso' else args.orig_labels_dir)
    
    image_files = sorted(list(images_dir.glob('*.nii.gz')))
    
    image_list_paths = [Path(p) for p in args.image_list] if args.image_list else []
    if image_list_paths:
        allowed_ids = set()
        for list_path in image_list_paths:
            if not list_path.exists():
                print(f"Warning: list file not found: {list_path}")
                continue
            ids = load_image_id_list(list_path)
            allowed_ids |= ids
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
        joined_lists = ", ".join(str(p) for p in image_list_paths)
        print(f"\nFiltered to {len(image_files)} images from lists: {joined_lists}")
    else:
        print(f"\nFound {len(image_files)} images to process")
    
    # Prepare metrics containers and CSV writer
    ensemble_metrics = {"dice": [], "cldice": []} if compute_metrics else None
    per_model_metrics = [{"dice": []} for _ in models] if compute_metrics else None
    csv_path = output_dir / 'per_image_metrics.csv' if compute_metrics else None
    
    # Track processed images for resume functionality
    processed_images = set()
    csv_mode = 'w'
    write_header = True
    if compute_metrics and args.resume and csv_path.exists():
        print(f"Resume mode enabled. Loading existing results from {csv_path}")
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            for row in reader:
                if row:
                    processed_images.add(row[0])  # image_id is first column
        print(f"Found {len(processed_images)} already processed images. Skipping them...")
        print()
        csv_mode = 'a'
        write_header = False
    
    writer = None
    csvfile_handle = None
    if compute_metrics:
        csvfile_handle = open(csv_path, csv_mode, newline='')
        writer = csv.writer(csvfile_handle)
        if write_header:
            header = (
                ['image_id', 'ensemble_dice', 'ensemble_cldice'] +
                [f'model_{i}_dice' for i in range(len(models))]
            )
            writer.writerow(header)
    
    try:
        for img_file in tqdm(image_files, desc="Processing images"):
        # Extract image number
            img_num = img_file.stem.split('.')[0].split('_')[1]
            img_id = f"image_{img_num}"
            
            # Skip if already processed (resume mode)
            if compute_metrics and img_id in processed_images:
                continue
            
            # Load image
            img_nifti = nib.load(str(img_file))
            image = img_nifti.get_fdata().astype(np.float32)
            # Temporarily disable lung masking
            lung_mask = np.ones_like(image, dtype=np.float32)
            
            # Predict each model sequentially, save, then fuse
            model_predictions = []
            for idx, model in enumerate(models):
                print(f"\n[Image {img_num}] Running model {idx+1}/{len(models)}")
                model_pred = predict_single_model(
                    model,
                    image,
                    device,
                    patch_size=tuple(args.patch_size),
                    overlap=args.overlap,
                    patch_batch_size=args.patch_batch_size,
                    use_amp=args.amp,
                )
                log_stats(f"Model {idx} pred stats", model_pred)
                model_predictions.append(model_pred)

                # Save per-model prediction
                m_dir = per_model_output_root / f'model_{idx}'
                m_dir.mkdir(parents=True, exist_ok=True)
                m_path = m_dir / f'pred_{img_num}.nii.gz'
                m_pred_to_save = prepare_for_saving(model_pred)
                m_nifti = nib.Nifti1Image(m_pred_to_save, img_nifti.affine, img_nifti.header)
                nib.save(m_nifti, str(m_path))

            # Fuse mean and STAPLE, already lung-masked
            prediction_mean = np.mean(model_predictions, axis=0)
            prediction_staple = fuse_predictions(
                model_predictions, method="staple", threshold=args.threshold
            )
            log_stats("Ensemble mean", prediction_mean)
            log_stats("Ensemble STAPLE", prediction_staple)

            # Save prediction (ensemble)
            mean_to_save = prepare_for_saving(prediction_mean)
            pred_nifti = nib.Nifti1Image(mean_to_save, img_nifti.affine, img_nifti.header)
            pred_path = mean_output_dir / f'pred_{img_num}.nii.gz'
            nib.save(pred_nifti, str(pred_path))

            staple_to_save = prepare_for_saving(prediction_staple)
            staple_nifti = nib.Nifti1Image(staple_to_save, img_nifti.affine, img_nifti.header)
            staple_path = staple_output_dir / f'pred_{img_num}.nii.gz'
            nib.save(staple_nifti, str(staple_path))
            
            if not compute_metrics:
                continue
            
            # Calculate Dice score if label exists
            label_file = labels_dir / f'label_{img_num}.nii.gz'
            if label_file.exists():
                label = load_label_resampled(label_file, img_file, image.shape)
                
                label_skeleton = None
                ensemble_dice, ensemble_cldice, label_skeleton = calculate_metrics(
                    prediction_mean, label, threshold=args.threshold, target_skeleton=None
                )
                ensemble_metrics["dice"].append(ensemble_dice)
                ensemble_metrics["cldice"].append(ensemble_cldice)
                print(
                    f"  Image {img_num}: Ensemble Dice = {ensemble_dice:.4f}, "
                    f"clDice = {ensemble_cldice:.4f}"
                )
                
                row = [img_id, ensemble_dice, ensemble_cldice]
                model_dice_vals = []
                for model_idx, model_pred in enumerate(model_predictions):
                    model_dice, _, _ = calculate_metrics(
                        model_pred, label,
                        threshold=args.threshold,
                        target_skeleton=None,
                        compute_cldice=False
                    )
                    per_model_metrics[model_idx]["dice"].append(model_dice)
                    model_dice_vals.append(model_dice)
                    print(
                        f"    Model {model_idx}: Dice = {model_dice:.4f}"
                    )
                row.extend(model_dice_vals)
                writer.writerow(row)
                csvfile_handle.flush()
    finally:
        if csvfile_handle is not None:
            csvfile_handle.close()
    
    # Print summary
    if compute_metrics and ensemble_metrics["dice"]:
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
            
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
            ax.bar([len(per_model_metrics)], [cldice_mean], yerr=[cldice_std if len(ensemble_metrics["cldice"]) > 1 else 0.0],
                   capsize=5, color='seagreen')
            ax.set_ylabel('clDice Score')
            ax.set_title('Ensemble clDice Score')
            ax.set_xticks([len(per_model_metrics)])
            ax.set_xticklabels(['Ensemble'])
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            fig.tight_layout()
            
            cldice_plot_path = output_dir / 'cldice_scores_bar.png'
            fig.savefig(cldice_plot_path, dpi=300)
            plt.close(fig)
            print(f"clDice bar plot saved to: {cldice_plot_path}")
    
    print(f"\nPredictions saved to: {output_dir}")
    print(f"Mean outputs:    {mean_output_dir}")
    print(f"STAPLE outputs:  {staple_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict using ensemble of 3D U-Nets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument(
        '--checkpoint_template',
        type=str,
        default=(
            '/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/'
            'Dataset001_nnunet/seed_{seed}/Dataset001_nnunet/'
            'nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
        ),
        help="Template path to checkpoints; must include '{seed}' placeholder (1..num_models).",
    )
    parser.add_argument('--num_models', type=int, default=10,
                        help='Number of models in the ensemble')
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Fold index used within each seed directory (e.g., fold_0)',
    )
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory (unused for labels; kept for compatibility)')
    parser.add_argument(
        '--images_dir',
        type=str,
        default='./data_preco/images_iso',
        help='Path to isotropic input images (NIfTI).',
    )
    parser.add_argument(
        '--orig_images_dir',
        type=str,
        default='./data/images',
        help='Path to original-resolution images (NIfTI).',
    )
    parser.add_argument(
        '--image_source',
        type=str,
        choices=['iso', 'orig'],
        default='iso',
        help="Select which images to use: 'iso' (default, data_preproc/images_iso) or 'orig' (data/images).",
    )
    parser.add_argument(
        '--labels_dir',
        type=str,
        default='./data_preco/labels_iso',
        help='Path to ground-truth labels (NIfTI) used when --image_source=iso.',
    )
    parser.add_argument(
        '--orig_labels_dir',
        type=str,
        default='./data/label',
        help='Path to ground-truth labels for original-resolution images.',
    )
    parser.add_argument(
        '--mean_output_dir',
        type=str,
        default='/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/mean',
        help='Path to save mean-averaged ensemble predictions',
    )
    parser.add_argument(
        '--staple_output_dir',
        type=str,
        default='/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/staple',
        help='Path to save STAPLE-fused ensemble predictions',
    )
    parser.add_argument(
        '--lung_mask_dir',
        type=str,
        default='data/lung_mask',
        help='Directory containing lung masks named lung_mask_XXX.nii.gz',
    )
    parser.add_argument('--image_list', type=str, action='append', default=None,
                        help='Optional text file(s) with image ids to process; pass multiple times for train/test/all')
    
    # Inference parameters
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Patch size for inference (D H W)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio between patches (0-1)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--patch_batch_size', type=int, default=4,
                        help='Number of patches to forward simultaneously during inference (raise to use more GPU memory)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed-precision (autocast) inference on CUDA')
    parser.add_argument('--per_model_output_root', type=str, default='./data/unet_prediction',
                        help='Base directory for per-model predictions (default: ./data/unet_prediction)')
    parser.add_argument('--binarize_saved_outputs', action='store_true', default=True,
                        help='Binarize saved predictions using --threshold (default: on)')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip Dice/clDice computation and CSV logging')
    parser.add_argument('--resume', action='store_true',
                        help='Resume prediction from where it was interrupted (skip already processed images)')
    
    args = parser.parse_args()
    
    predict_on_dataset(args)


if __name__ == '__main__':
    main()
