"""Preprocess volumes: normalize, resample to 1mm iso, crop ROI, and save metadata.
python preproc_prepare_rois.py --images_dir data/images --labels_dir data/labels --output_root data_preproc --target_spacing 1.0 --margin_mm 8.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

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


def robust_normalize(volume: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Clip to 1/99 percentiles (on mask if provided) and rescale to [0,1]."""
    work = volume[mask] if mask is not None and mask.any() else volume
    p1, p99 = np.percentile(work, [1, 99])
    if p99 <= p1:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, p1, p99)
    norm = (clipped - p1) / (p99 - p1)
    return norm.astype(np.float32, copy=False)


def resample_to_iso(volume: np.ndarray, spacing: Tuple[float, float, float], target: float, order: int) -> np.ndarray:
    factors = tuple(s / target for s in spacing)
    return zoom(volume, zoom=factors, order=order)


def bbox_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return mins, maxs


def crop_with_margin(volume: np.ndarray, mins: np.ndarray, maxs: np.ndarray, margin_vox: int) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    start = np.maximum(mins - margin_vox, 0)
    end = np.minimum(maxs + margin_vox + 1, volume.shape)
    slices = tuple(slice(s, e) for s, e in zip(start, end))
    return volume[slices], slices


def find_label(label_dir: Path, case_id: str) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        candidate = label_dir / f"label_{case_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize, resample to 1mm, crop ROI, and save metadata.")
    parser.add_argument("--images_dir", type=Path, default=Path("data/images"))
    parser.add_argument("--labels_dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--output_root", type=Path, default=Path("data_preproc"))
    parser.add_argument("--split_file", type=Path, default=None, help="Optional list of IDs to process (without extension).")
    parser.add_argument("--target_spacing", type=float, default=1.0, help="Isotropic spacing in mm.")
    parser.add_argument("--margin_mm", type=float, default=8.0, help="Margin around label bbox in mm.")
    args = parser.parse_args()

    ids: set[str] | None = None
    if args.split_file:
        ids = {line.strip() for line in args.split_file.read_text().splitlines() if line.strip()}

    out_iso_img = args.output_root / "images_iso"
    out_iso_lbl = args.output_root / "labels_iso"
    out_roi_img = args.output_root / "images_roi"
    out_roi_lbl = args.output_root / "labels_roi"
    out_meta = args.output_root / "meta"
    for d in [out_iso_img, out_iso_lbl, out_roi_img, out_roi_lbl, out_meta]:
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
        lbl_data = lbl.get_fdata()
        lbl_mask = lbl_data > 0

        norm = robust_normalize(img_data, lbl_mask)

        img_iso = resample_to_iso(norm, spacing, target=args.target_spacing, order=1)
        lbl_iso = resample_to_iso(lbl_mask.astype(np.float32), spacing, target=args.target_spacing, order=0)

        target_spacing = args.target_spacing
        margin_vox = int(round(args.margin_mm / target_spacing))

        bbox = bbox_from_mask(lbl_iso > 0.5)
        if bbox is None:
            # No foreground, keep full volume.
            roi_img = img_iso
            roi_lbl = lbl_iso
            slices = (slice(0, img_iso.shape[0]), slice(0, img_iso.shape[1]), slice(0, img_iso.shape[2]))
        else:
            mins, maxs = bbox
            roi_img, slices = crop_with_margin(img_iso, mins, maxs, margin_vox)
            roi_lbl = crop_with_margin(lbl_iso, mins, maxs, margin_vox)[0]

        # Save iso volumes
        affine_iso = np.diag([target_spacing, target_spacing, target_spacing, 1.0])
        iso_header = img.header.copy()
        iso_header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(img_iso.astype(np.float32, copy=False), affine_iso, iso_header), str(out_iso_img / f"image_{case_id}.nii.gz"))

        lbl_header = lbl.header.copy()
        lbl_header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(lbl_iso.astype(np.uint8), affine_iso, lbl_header), str(out_iso_lbl / f"label_{case_id}.nii.gz"))

        # Save ROI volumes
        nib.save(nib.Nifti1Image(roi_img.astype(np.float32, copy=False), affine_iso, iso_header), str(out_roi_img / f"image_{case_id}.nii.gz"))
        nib.save(nib.Nifti1Image(roi_lbl.astype(np.uint8), affine_iso, lbl_header), str(out_roi_lbl / f"label_{case_id}.nii.gz"))

        # Metadata
        meta = {
            "case_id": case_id,
            "orig_spacing": [float(s) for s in spacing],
            "target_spacing": float(target_spacing),
            "margin_mm": float(args.margin_mm),
            "roi_slices": [[int(s.start), int(s.stop)] for s in slices],
            "orig_shape": [int(v) for v in img_data.shape],
            "iso_shape": [int(v) for v in img_iso.shape],
        }
        meta_path = out_meta / f"{case_id}_roi.json"
        meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
