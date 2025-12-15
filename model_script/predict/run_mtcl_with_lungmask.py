"""
Pipeline to (1) generate lung masks with TotalSegmentator and (2) run MTCL
prediction masked to lungs.

Example (data root at "path"):
    ./noisy_seg_env/bin/python -m model_script.predict.run_mtcl_with_lungmask \
        --data_dir path/data \
        --checkpoint path/checkpoints/leader_final.pth \
        --output_dir path/results/mtcl_lungmasked

If --mask_dir is omitted, masks are written to a temporary folder that is
deleted after the run.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from model_script.predict import predict_mtcl_model as mtcl_pred
from model_script.predict import total_segment_lung_mask as tslm
from model_script.train import mtcl as mtcl_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TotalSegmentator lung masks then run MTCL prediction using them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=Path("./data"), help="Root containing images/ and labels/")
    parser.add_argument(
        "--ids_file",
        type=Path,
        default=None,
        help="Optional text file with image ids (e.g., image_001). If omitted, all NIfTI files in data_dir/images are used.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Path to MTCL checkpoint (.pth). If omitted, the latest file in {mtcl_pred.DEFAULT_MTCL_CKPT_DIR} is used.",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="Where to write lung masks. Defaults to a temporary directory that is cleaned up after the run.",
    )
    parser.add_argument("--overwrite_masks", action="store_true", help="Regenerate masks even if they already exist.")
    parser.add_argument(
        "--keep_intermediate_masks",
        action="store_true",
        help="Keep per-lobe masks emitted by TotalSegmentator next to the binary mask (for debugging).",
    )
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Sliding window patch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio between sliding windows")
    parser.add_argument("--patch_batch_size", type=int, default=8, help="Number of patches per forward pass")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional directory to save NIfTI predictions (pred_<id>.nii.gz)",
    )
    parser.add_argument("--save_probs", action="store_true", help="If set, save probability maps instead of binary masks")
    parser.add_argument("--csv_path", type=Path, default=None, help="Optional CSV path for per-image metrics")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device string for torch")
    return parser.parse_args()


def generate_lung_masks(
    ids: Iterable[str],
    images_dir: Path,
    mask_dir: Path,
    *,
    overwrite: bool,
    keep_intermediate: bool,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device not available; TotalSegmentator lung masking requires GPU.")

    mask_dir.mkdir(parents=True, exist_ok=True)

    for cid in tqdm(ids, desc="Lung masks"):
        img_path = mtcl_train.find_matching_file(images_dir, cid)
        if img_path is None:
            print(f"[skip] {cid}: image not found under {images_dir}")
            continue
        tslm.process_image(
            case_id=cid,
            image_path=img_path,
            output_dir=mask_dir,
            device="gpu",
            overwrite=overwrite,
            keep_intermediate=keep_intermediate,
        )


def run_prediction_with_masks(
    model: mtcl_pred.UNet3D,
    ids: Iterable[str],
    data_dir: Path,
    mask_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    overlap: float,
    patch_batch_size: int,
    threshold: float,
    save_dir: Path | None,
    save_probs: bool,
    csv_path: Path | None,
):
    dice_scores = []
    cldice_scores = []
    rows = []

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    for cid in tqdm(ids, desc="Predict + evaluate (lung-masked)"):
        img_path = mtcl_train.find_matching_file(data_dir / "images", cid)
        lbl_path = mtcl_train.find_matching_file(data_dir / "labels", cid)
        if img_path is None or lbl_path is None:
            print(f"[skip] Missing image/label for {cid}")
            continue

        image, affine = mtcl_train.load_nifti_array(img_path)
        label, _ = mtcl_train.load_nifti_array(lbl_path)
        image = mtcl_train.normalize_ct(image)

        volume = torch.from_numpy(np.ascontiguousarray(image)[None, None]).to(device)
        probs = mtcl_train.sliding_window_predict(
            model,
            volume,
            patch_size=patch_size,
            device=device,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
        ).cpu().numpy()[0, 1]

        mask_path = mask_dir / tslm.mask_filename(cid)
        if not mask_path.exists():
            print(f"[warn] Lung mask not found for {cid} at {mask_path}; skipping mask application.")
            lung_mask = None
        else:
            lung_mask = nib.load(str(mask_path)).get_fdata()
            if lung_mask.shape != probs.shape:
                raise ValueError(
                    f"Lung mask shape {lung_mask.shape} does not match prediction shape {probs.shape} for {cid}"
                )
            probs = probs * (lung_mask > 0)

        dice, cldice, _ = mtcl_pred.calculate_metrics(
            probs,
            label,
            threshold=threshold,
            target_skeleton=None,
            compute_cldice=True,
        )

        dice_scores.append(dice)
        cldice_scores.append(cldice)
        rows.append((cid, dice, cldice))
        print(f"{cid}: Dice={dice:.4f}, clDice={cldice:.4f}")

        if save_dir is not None:
            to_save = probs if save_probs else (probs >= threshold).astype(np.uint8)
            nib.save(nib.Nifti1Image(to_save, affine), save_dir / f"pred_{cid}.nii.gz")

    if not rows:
        print("No samples evaluated.")
        return

    print("\n" + "=" * 50)
    print("MTCL EVALUATION SUMMARY (lung-masked)")
    print("=" * 50)
    dice_arr = np.array(dice_scores, dtype=np.float64)
    cldice_arr = np.array(cldice_scores, dtype=np.float64)
    print(f"Images evaluated: {len(rows)}")
    print(f"Dice   Mean ± Std: {dice_arr.mean():.4f} ± {dice_arr.std():.4f}")
    print(f"Dice   Min / Max:  {dice_arr.min():.4f} / {dice_arr.max():.4f}")
    print(f"clDice Mean ± Std: {cldice_arr.mean():.4f} ± {cldice_arr.std():.4f}")
    print(f"clDice Min / Max:  {cldice_arr.min():.4f} / {cldice_arr.max():.4f}")

    if csv_path is not None:
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "dice", "cldice"])
            writer.writerows(rows)
        print(f"Saved per-image metrics to {csv_path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = mtcl_pred.find_latest_mtcl_checkpoint(mtcl_pred.DEFAULT_MTCL_CKPT_DIR)
        print(f"[Auto] Using latest MTCL checkpoint: {ckpt_path}")

    model = mtcl_pred.load_mtcl_model(ckpt_path, device)
    if args.ids_file is not None:
        ids = list(mtcl_pred._iter_image_ids(args.ids_file))
    else:
        ids = mtcl_pred._list_image_ids_from_dir(args.data_dir / "images")
        print(f"[Auto] Using {len(ids)} ids from {args.data_dir / 'images'}")

    mask_ctx = (
        TemporaryDirectory()
        if args.mask_dir is None
        else nullcontext(args.mask_dir)
    )
    with mask_ctx as mask_root:
        mask_dir = Path(mask_root) if args.mask_dir is None else Path(mask_root)
        if args.mask_dir is None:
            print(f"[Masks] Using temporary directory: {mask_dir}")
        else:
            mask_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Masks] Using mask directory: {mask_dir}")

        generate_lung_masks(
            ids,
            args.data_dir / "images",
            mask_dir,
            overwrite=args.overwrite_masks,
            keep_intermediate=args.keep_intermediate_masks,
        )

        run_prediction_with_masks(
            model=model,
            ids=ids,
            data_dir=args.data_dir,
            mask_dir=mask_dir,
            patch_size=tuple(args.patch_size),
            device=device,
            overlap=args.overlap,
            patch_batch_size=args.patch_batch_size,
            threshold=args.threshold,
            save_dir=args.output_dir,
            save_probs=args.save_probs,
            csv_path=args.csv_path,
        )

    if args.mask_dir is None:
        print("[Masks] Temporary mask directory cleaned up.")
    else:
        print(f"[Masks] Masks persisted at {args.mask_dir}")


if __name__ == "__main__":
    main()
