"""Predict with a saved MTCL leader model and report Dice/clDice.
python model_script/predict/predict_mtcl_model.py --data_dir ./data --ids_file data/splits/test_list.txt --output_dir ./data/mtcl/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.unet_model import UNet3D
from model_script.train import mtcl as mtcl_train
from model_script.predict.predict_ensemble import calculate_metrics

DEFAULT_MTCL_CKPT_DIR = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_mtcl/mtcl_best.pth")


def _resolve_state_dict(raw_obj: dict) -> dict:
    """
    Extract the actual model state dict from common MTCL checkpoint layouts.
    Prefers teacher -> student -> model_state_dict -> state_dict.
    """
    if not isinstance(raw_obj, dict):
        return raw_obj
    for key in ("teacher", "student", "model_state_dict", "state_dict"):
        if key in raw_obj and isinstance(raw_obj[key], dict):
            return raw_obj[key]
    return raw_obj


def load_mtcl_model(ckpt_path: Path, device: torch.device) -> UNet3D:
    """
    Load a trained MTCL leader (teacher) model from checkpoint.
    Expands legacy 1-channel heads to 2-channel outputs.
    """
    raw = torch.load(ckpt_path, map_location=device)
    state = _resolve_state_dict(raw)

    model = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
    model_state = model.state_dict()
    filtered_state = {}

    head_w = state.get("outc.conv.weight") if isinstance(state, dict) else None
    head_b = state.get("outc.conv.bias") if isinstance(state, dict) else None
    if head_w is not None and head_b is not None:
        if head_w.shape[0] == 1 and model_state["outc.conv.weight"].shape[0] == 2:
            new_w = model_state["outc.conv.weight"].clone()
            new_b = model_state["outc.conv.bias"].clone()
            new_w.zero_()
            new_b.zero_()
            new_w[1] = head_w[0]
            new_b[1] = head_b[0]
            filtered_state["outc.conv.weight"] = new_w
            filtered_state["outc.conv.bias"] = new_b
            print("[Load] Expanded 1->2 channel head from checkpoint.")

    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")

    for k, v in state.items():
        if k in filtered_state:
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[Load] Missing params: {missing}")
    if unexpected:
        print(f"[Load] Unexpected params: {unexpected}")
    model.eval()
    return model


def _iter_image_ids(list_path: Path) -> Iterable[str]:
    return mtcl_train.read_id_list(list_path)


def _list_image_ids_from_dir(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    ids = {
        mtcl_train._normalize_image_id(p.name)
        for p in images_dir.glob("*.nii*")
    }
    if not ids:
        raise ValueError(f"No NIfTI files found in {images_dir}")
    return sorted(ids)


def find_latest_mtcl_checkpoint(ckpt_dir: Path) -> Path:
    """
    Pick the latest MTCL checkpoint, preferring leader_final.pth when present.
    Accepts a directory or direct .pth file path.
    """
    if ckpt_dir.is_file():
        return ckpt_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint path is not a directory: {ckpt_dir}")

    leader_final = ckpt_dir / "leader_final.pth"
    if leader_final.exists():
        return leader_final

    candidates = list(ckpt_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found under {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


@torch.no_grad()
def run_prediction(
    model: UNet3D,
    ids: Iterable[str],
    data_dir: Path,
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

    for cid in tqdm(ids, desc="Predict + evaluate"):
        img_path = mtcl_train.find_matching_file(data_dir / "images", cid)
        lbl_path = mtcl_train.find_matching_file(data_dir / "labels", cid)
        if img_path is None or lbl_path is None:
            print(f"[Skip] Missing image/label for {cid}")
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

        dice, cldice, _ = calculate_metrics(
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
    print("MTCL EVALUATION SUMMARY")
    print("=" * 50)
    dice_arr = np.array(dice_scores, dtype=np.float64)
    cldice_arr = np.array(cldice_scores, dtype=np.float64)
    print(f"Images evaluated: {len(rows)}")
    print(f"Dice   Mean ± Std: {dice_arr.mean():.4f} ± {dice_arr.std():.4f}")
    print(f"Dice   Min / Max:  {dice_arr.min():.4f} / {dice_arr.max():.4f}")
    print(f"clDice Mean ± Std: {cldice_arr.mean():.4f} ± {cldice_arr.std():.4f}")
    print(f"clDice Min / Max:  {cldice_arr.min():.4f} / {cldice_arr.max():.4f}")

    if csv_path is not None:
        import csv

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "dice", "cldice"])
            writer.writerows(rows)
        print(f"Saved per-image metrics to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict with a saved MTCL leader model and compute Dice/clDice.",
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
        default="/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_mtcl/mtcl_best.pth",
        help=f"Path to MTCL checkpoint (.pth). If omitted, the latest file in {DEFAULT_MTCL_CKPT_DIR} is used.",
    )
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Sliding window patch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio between sliding windows")
    parser.add_argument("--patch_batch_size", type=int, default=48, help="Number of patches per forward pass")
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        try:
            ckpt_path = find_latest_mtcl_checkpoint(DEFAULT_MTCL_CKPT_DIR)
            print(f"[Auto] Using latest MTCL checkpoint: {ckpt_path}")
        except FileNotFoundError as exc:
            raise SystemExit(
                f"No checkpoint provided and none found under {DEFAULT_MTCL_CKPT_DIR}. "
                f"Pass --checkpoint to specify a file."
            ) from exc

    model = load_mtcl_model(ckpt_path, device)
    if args.ids_file is not None:
        ids = list(_iter_image_ids(args.ids_file))
    else:
        ids = _list_image_ids_from_dir(args.data_dir / "images")
        print(f"[Auto] Using {len(ids)} ids from {args.data_dir / 'images'}")
    run_prediction(
        model=model,
        ids=ids,
        data_dir=args.data_dir,
        patch_size=tuple(args.patch_size),
        device=device,
        overlap=args.overlap,
        patch_batch_size=args.patch_batch_size,
        threshold=args.threshold,
        save_dir=args.output_dir,
        save_probs=args.save_probs,
        csv_path=args.csv_path,
    )


if __name__ == "__main__":
    main()
