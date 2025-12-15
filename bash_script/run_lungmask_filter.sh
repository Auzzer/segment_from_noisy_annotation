"""
Post-process saved predictions by masking out non-lung voxels using lungmask.

Example:
    python -m model_script.predict.lung_mask \
        --images_dir /path/to/images \
        --pred_dir /path/to/unet_prediction/ensemble \
        --output_dir /path/to/unet_prediction/ensemble_lungmasked \
        --binarize_output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("PyTorch is required for lungmask GPU checks.") from exc

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "SimpleITK is required for lungmask filtering. Install it with `pip install SimpleITK`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask vessel predictions to the lung region using lungmask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/images"),
        help="Directory containing the CT volumes used for prediction",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("data/unet_prediction/ensemble"),
        help="Directory with saved vessel predictions (NIfTI)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write lung-masked predictions (defaults to pred_dir/lungmask_filtered)",
    )
    parser.add_argument(
        "--pred_glob",
        type=str,
        default="pred_*.nii.gz",
        help="Glob pattern used to pick prediction files inside pred_dir",
    )
    parser.add_argument(
        "--id_list",
        type=Path,
        default=None,
        help="Optional text file with image ids to process (one per line)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force lungmask to run on CPU even if CUDA is available",
    )
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Fail if CUDA is unavailable (useful to guarantee GPU execution)",
    )
    parser.add_argument(
        "--binarize_output",
        action="store_true",
        help="After masking, threshold predictions to 0/1",
    )
    parser.add_argument(
        "--binarize_threshold",
        type=float,
        default=0.5,
        help="Threshold used when binarizing outputs",
    )
    return parser.parse_args()


def load_lungmask_model():
    """
    Load lungmask and return a tuple describing the backend plus the callable model.
    Prefers LMInferer (default U-Net) and falls back to mask.get_model().
    """
    try:
        from lungmask import LMInferer
        inferer = LMInferer()
        return ("inferer", inferer)
    except Exception:
        try:
            from lungmask import mask as lungmask_mask
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise SystemExit(
                "The 'lungmask' package is required. Install it with `pip install lungmask SimpleITK`."
            ) from exc
        try:
            model = lungmask_mask.get_model()
        except TypeError:
            # Some versions require the modelname argument explicitly
            try:
                model = lungmask_mask.get_model()
            except TypeError:
                # Fall back to positional modelname only
                model = lungmask_mask.get_model()
        return ("mask_module", (lungmask_mask, model))


def strip_prediction_id(pred_path: Path) -> str:
    """Derive image id from a prediction filename."""
    name = pred_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.startswith("pred_"):
        name = name[len("pred_") :]
    return name


def load_id_whitelist(list_path: Path | None) -> Set[str] | None:
    if list_path is None:
        return None
    ids: Set[str] = set()
    with open(list_path, "r") as f:
        for line in f:
            entry = line.strip()
            if entry:
                ids.add(entry if entry.startswith("image_") else f"image_{entry}")
    return ids


def find_image(images_dir: Path, case_id: str) -> Path | None:
    """Find the matching CT image for a prediction id."""
    candidates = [
        images_dir / f"{case_id}.nii.gz",
        images_dir / f"{case_id}.nii",
    ]
    if not case_id.startswith("image_"):
        candidates.extend(
            [
                images_dir / f"image_{case_id}.nii.gz",
                images_dir / f"image_{case_id}.nii",
            ]
        )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def resample_mask_to_prediction(mask_np: np.ndarray, image_ct: sitk.Image, pred_img: sitk.Image) -> np.ndarray:
    """Resample a lung mask (aligned to the CT) onto the prediction geometry."""
    mask_img = sitk.GetImageFromArray(mask_np.astype(np.float32))
    mask_img.CopyInformation(image_ct)
    resampled = sitk.Resample(
        mask_img,
        pred_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
    )
    return sitk.GetArrayFromImage(resampled)


def mask_prediction(
    pred_path: Path,
    image_path: Path,
    output_path: Path,
    lungmask_backend,
    *,
    force_cpu: bool,
    binarize_output: bool,
    binarize_threshold: float,
) -> None:
    image_ct = sitk.ReadImage(str(image_path))
    pred_img = sitk.ReadImage(str(pred_path))

    backend_type = lungmask_backend[0]
    if backend_type == "inferer":
        inferer = lungmask_backend[1]
        lung_mask_np = inferer.apply(image_ct, force_cpu=force_cpu)
    else:
        lungmask_module, lungmask_model = lungmask_backend[1]
        lung_mask_np = lungmask_module.apply(image_ct, model=lungmask_model, force_cpu=force_cpu)
    lung_mask_np = (np.asarray(lung_mask_np) > 0).astype(np.float32)

    pred_np = sitk.GetArrayFromImage(pred_img).astype(np.float32)
    if pred_np.shape != lung_mask_np.shape:
        lung_mask_np = resample_mask_to_prediction(lung_mask_np, image_ct, pred_img)

    filtered = pred_np * lung_mask_np

    if binarize_output:
        filtered = (filtered >= binarize_threshold).astype(pred_np.dtype, copy=False)

    out_img = sitk.GetImageFromArray(filtered.astype(pred_np.dtype, copy=False))
    out_img.CopyInformation(pred_img)
    sitk.WriteImage(out_img, str(output_path))


def collect_predictions(pred_dir: Path, glob_pat: str, whitelist: Set[str] | None) -> Iterable[Path]:
    for pred_path in sorted(pred_dir.glob(glob_pat)):
        case_id = strip_prediction_id(pred_path)
        normalized = case_id if case_id.startswith("image_") else f"image_{case_id}"
        if whitelist is not None and normalized not in whitelist:
            continue
        yield pred_path


def main():
    args = parse_args()

    if args.force_cpu and args.require_cuda:
        raise SystemExit("Use either --force_cpu or --require_cuda, not both.")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested via --require_cuda, but no GPU was detected.")

    pred_dir = args.pred_dir
    if not pred_dir.exists():
        raise SystemExit(f"Prediction directory not found: {pred_dir}")

    output_dir = args.output_dir or pred_dir / "lungmask_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    whitelist = load_id_whitelist(args.id_list)
    preds = list(collect_predictions(pred_dir, args.pred_glob, whitelist))
    if not preds:
        raise SystemExit(f"No prediction files found in {pred_dir} matching pattern '{args.pred_glob}'")

    lungmask_backend = load_lungmask_model()

    skipped_missing = []
    skipped_existing = []
    processed = 0

    for pred_path in tqdm(preds, desc="Applying lungmask"):
        case_id = strip_prediction_id(pred_path)
        case_id_norm = case_id if case_id.startswith("image_") else f"image_{case_id}"
        img_path = find_image(args.images_dir, case_id_norm)
        if img_path is None:
            skipped_missing.append(case_id_norm)
            continue

        out_path = output_dir / pred_path.name
        if out_path.exists() and not args.overwrite:
            skipped_existing.append(case_id_norm)
            continue

        mask_prediction(
            pred_path,
            img_path,
            out_path,
            lungmask_backend,
            force_cpu=args.force_cpu,
            binarize_output=args.binarize_output,
            binarize_threshold=args.binarize_threshold,
        )
        processed += 1

    print(f"\nDone. Wrote {processed} file(s) to {output_dir}")
    if skipped_existing:
        print(f"Skipped {len(skipped_existing)} already-present file(s); use --overwrite to re-run them.")
    if skipped_missing:
        print("Missing CT volumes for the following ids (skipped):")
        for cid in skipped_missing:
            print(f"  - {cid}")


if __name__ == "__main__":
    main()
