"""Preprocess volumes: normalize (CT HU clip + min-max), resample to 1mm iso, save full volumes.

Example:
  python -m model_script.preproc.preproc_normalize \
    --images_dir data/images --labels_dir data/labels \
    --output_root data_preproc --target_spacing 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def parse_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("image_", "").replace("label_", "")


def ct_normalize(volume: np.ndarray) -> np.ndarray:
    """Clip HU to [-1000, 400] then min-max to [0,1]."""
    clipped = np.clip(volume, -1000.0, 400.0)
    norm = (clipped + 1000.0) / 1400.0
    return norm.astype(np.float32, copy=False)


def resample_to_iso(volume: np.ndarray, spacing: Tuple[float, float, float], target: float, order: int) -> np.ndarray:
    factors = tuple(s / target for s in spacing)
    return zoom(volume, zoom=factors, order=order)


def find_label(label_dir: Path, case_id: str) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        candidate = label_dir / f"label_{case_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize CT, resample to 1mm iso, and save full volumes.")
    parser.add_argument("--images_dir", type=Path, default=Path("data/images"))
    parser.add_argument("--labels_dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--output_root", type=Path, default=Path("data_preproc"))
    parser.add_argument("--split_file", type=Path, default=None, help="Optional list of IDs to process (without extension).")
    parser.add_argument("--target_spacing", type=float, default=1.0, help="Isotropic spacing in mm.")
    args = parser.parse_args()

    ids: set[str] | None = None
    if args.split_file:
        ids = {line.strip() for line in args.split_file.read_text().splitlines() if line.strip()}

    out_iso_img = args.output_root / "images_iso"
    out_iso_lbl = args.output_root / "labels_iso"
    for d in [out_iso_img, out_iso_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in args.images_dir.iterdir() if p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz"))
    for img_path in tqdm(image_files, desc="Preprocess"):
        case_id = parse_id(img_path)
        if ids and case_id not in ids:
            continue

        lbl_path = find_label(args.labels_dir, case_id)
        if lbl_path is None:
            print(f"Skipping {case_id}: no matching label in {args.labels_dir}")
            continue

        img = nib.load(str(img_path))
        lbl = nib.load(str(lbl_path))
        spacing = img.header.get_zooms()[:3]

        img_data = img.get_fdata().astype(np.float32, copy=False)
        lbl_data = lbl.get_fdata().astype(np.float32, copy=False)
        lbl_mask = lbl_data > 0

        norm = ct_normalize(img_data)

        img_iso = resample_to_iso(norm, spacing, target=args.target_spacing, order=1)
        lbl_iso = resample_to_iso(lbl_mask.astype(np.float32), spacing, target=args.target_spacing, order=0)

        target_spacing = args.target_spacing

        # Save iso volumes
        affine_iso = np.diag([target_spacing, target_spacing, target_spacing, 1.0])
        iso_header = img.header.copy()
        iso_header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(img_iso.astype(np.float32, copy=False), affine_iso, iso_header), str(out_iso_img / f"image_{case_id}.nii.gz"))

        lbl_header = lbl.header.copy()
        lbl_header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(lbl_iso.astype(np.uint8), affine_iso, lbl_header), str(out_iso_lbl / f"label_{case_id}.nii.gz"))


if __name__ == "__main__":
    main()
