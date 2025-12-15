"""Run cucim.skimage Frangi filter on a NIfTI volume.
Usage:
export CUDA_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python frangi_gpu.py \
  --input data/images/image_001.nii.gz \
  --output_dir ./results/tmp/cucim_out \
  --sigmas 1,2,3
  
python frangi_gpu.py \
  --input data_preproc/images_roi/image_001.nii.gz \
  --label data_preproc/labels_roi/label_001.nii.gz \
  --output_dir ./results/tmp/frangi_roi \
  --sigmas 0.6,0.8,1.2 \
  --alpha 0.5 --beta 0.5 --device 0 \
  --threshold 0.2    # optional: saves _label, computes Dice if label provided
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cupy as cp
from cucim.skimage.filters import frangi
import nibabel as nib
import numpy as np


def parse_sigmas(sigmas_arg: str | Iterable[float]) -> List[float]:
    if isinstance(sigmas_arg, str):
        return [float(s.strip()) for s in sigmas_arg.split(",") if s.strip()]
    return [float(s) for s in sigmas_arg]


def sigmas_mm_to_vox(
    sigmas_mm: Iterable[float], spacing: Iterable[float]
) -> List[float | tuple[float, float, float]]:
    spacing_arr = np.array(list(spacing), dtype=float)
    isotropic = np.allclose(spacing_arr, spacing_arr[0])

    def convert(val: float) -> float | tuple[float, float, float]:
        if isotropic:
            return float(val / spacing_arr[0])
        return tuple(float(val / s) for s in spacing_arr)

    return [convert(v) for v in sigmas_mm]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Apply cucim Frangi filter (GPU) to a NIfTI volume."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input NIfTI.")
    parser.add_argument("--label", type=Path, default=None, help="Optional label NIfTI for Dice computation.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--sigmas",
        type=str,
        default="1,2,3",
        help="Comma-separated Gaussian sigmas (e.g. 1,2,3).",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Frangi alpha.")
    parser.add_argument("--beta", type=float, default=0.5, help="Frangi beta.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Structuredness term for Frangi (default: auto).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, also save a binary labelmap (>= threshold) as uint8.",
    )
    parser.add_argument(
        "--white_ridges",
        action="store_true",
        help="Detect bright ridges instead of dark ridges.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="nearest",
        choices=["nearest", "reflect", "mirror", "wrap", "constant"],
        help="Border handling mode.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id.")

    args = parser.parse_args(argv)
    sigmas = parse_sigmas(args.sigmas)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(args.input))
    data = img.get_fdata().astype(np.float32, copy=False)
    spacing = img.header.get_zooms()[: data.ndim]

    label_data = None
    if args.label:
        lbl_img = nib.load(str(args.label))
        label_data = lbl_img.get_fdata()

    with cp.cuda.Device(args.device):
        x = cp.asarray(data, dtype=cp.float32)
        sigmas_vox = sigmas_mm_to_vox(sigmas, spacing)
        filtered = frangi(
            x,
            sigmas=sigmas_vox,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            black_ridges=not args.white_ridges,
            mode=args.mode,
        )
        result = cp.asnumpy(filtered)
        #cp.get_default_memory_pool().free_all_blocks()

    img.header.set_data_dtype(np.float32)
    out_name = f"frangi_{args.input.name}"
    out_path = output_dir / out_name
    nib.save(nib.Nifti1Image(result, img.affine, img.header), str(out_path))

    dice_value = None
    if args.threshold is not None:
        label = (result >= args.threshold).astype(np.uint8)
        img.header.set_data_dtype(np.uint8)
        if out_name.endswith(".nii.gz"):
            label_name = out_name[: -len(".nii.gz")] + "_label.nii.gz"
        elif out_name.endswith(".nii"):
            label_name = out_name[: -len(".nii")] + "_label.nii"
        else:
            label_name = out_name + "_label"
        label_path = output_dir / label_name
        nib.save(nib.Nifti1Image(label, img.affine, img.header), str(label_path))

        if label_data is not None:
            gt = (label_data > 0).astype(np.uint8)
            pred = label
            inter = np.logical_and(pred, gt).sum()
            denom = pred.sum() + gt.sum()
            dice_value = 2 * inter / denom if denom > 0 else 1.0

    return dice_value

if __name__ == "__main__":
    main()
