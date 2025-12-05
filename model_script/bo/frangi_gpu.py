"""Run cucim.skimage Frangi filter on a NIfTI volume.
Usage:
export CUDA_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python frangi_gpu.py \
  --input data/images/image_001.nii.gz \
  --sigmas 1,2,3

python frangi_gpu.py \
  --input data/images/image_001.nii.gz \
  --label data/labels/label_001.nii.gz \
  --sigmas 0.6,0.8,1.2 \
  --alpha 0.5 --beta 0.5 --device 0 \
  --threshold 0.2 \
  --save_outputs --output_dir ./results/tmp/frangi_roi   # only if you want NIfTI writes

By default no files are written; Dice is still computed if a label+threshold are provided.
NIfTI outputs require --save_outputs; when Dice is computed a small best-params JSON is updated (see --best_file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

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


def update_best_params(
    best_path: Path, params: dict[str, object], dice: float
) -> tuple[bool, float | None]:
    """Update a JSON file with the best Dice/params seen so far.

    Returns a tuple (updated, previous_best_dice).
    """
    best_path.parent.mkdir(parents=True, exist_ok=True)

    prev_best = None
    if best_path.exists():
        try:
            prev_data = json.loads(best_path.read_text())
            prev_best = float(prev_data.get("dice"))
        except Exception:
            prev_best = None

    improved = prev_best is None or dice >= prev_best
    if improved:
        payload = {"params": params, "dice": float(dice)}
        best_path.write_text(json.dumps(payload, indent=2))

    return improved, prev_best


def run_frangi_array(
    image: np.ndarray,
    spacing: Sequence[float],
    *,
    sigmas: Sequence[float],
    alpha: float,
    beta: float,
    gamma: float | None,
    threshold: float | None,
    label_data: np.ndarray | None,
    device: int,
    white_ridges: bool = False,
    mode: str = "nearest",
    save_outputs: bool = True,
    output_dir: Path | None = None,
    output_basename: str | None = None,
    affine: np.ndarray | None = None,
    header: nib.Nifti1Header | None = None,
) -> float | None:
    """Run Frangi on an in-memory array. Optionally saves outputs, returns Dice."""
    if save_outputs and output_dir is None:
        raise ValueError("output_dir is required when save_outputs=True")
    if save_outputs and affine is None:
        raise ValueError("affine is required when save_outputs=True")

    output_path = None
    label_path = None
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        base = output_basename or "frangi_output.nii.gz"
        output_path = output_dir / base

    spacing_tuple = tuple(float(s) for s in spacing)

    with cp.cuda.Device(device):
        x = cp.asarray(image, dtype=cp.float32)
        sigmas_vox = sigmas_mm_to_vox(sigmas, spacing_tuple)
        filtered = frangi(
            x,
            sigmas=sigmas_vox,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=not white_ridges,
            mode=mode,
        )

        dice_value = None

        need_mask = threshold is not None and (save_outputs or label_data is not None)
        mask_cpu = None
        if need_mask:
            mask = filtered >= threshold
            if label_data is not None:
                gt = cp.asarray(label_data > 0, dtype=cp.uint8)
                inter = cp.logical_and(mask, gt).sum(dtype=cp.int64)
                denom = mask.sum(dtype=cp.int64) + gt.sum(dtype=cp.int64)
                dice_value = (
                    float((2.0 * inter / denom).item()) if int(denom) > 0 else 1.0
                )
            if save_outputs:
                mask_cpu = cp.asnumpy(mask.astype(np.uint8))

        if save_outputs:
            result_cpu = cp.asnumpy(filtered)
            hdr = header.copy() if header is not None else None
            if hdr is not None:
                hdr.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(result_cpu, affine, hdr), str(output_path))

            if mask_cpu is not None:
                hdr_mask = header.copy() if header is not None else None
                if hdr_mask is not None:
                    hdr_mask.set_data_dtype(np.uint8)
                if output_path.name.endswith(".nii.gz"):
                    label_name = output_path.name[: -len(".nii.gz")] + "_label.nii.gz"
                elif output_path.name.endswith(".nii"):
                    label_name = output_path.name[: -len(".nii")] + "_label.nii"
                else:
                    label_name = output_path.name + "_label"
                label_path = output_dir / label_name
                nib.save(nib.Nifti1Image(mask_cpu, affine, hdr_mask), str(label_path))

    return dice_value


def run_frangi_file(
    input_path: Path,
    label_path: Path | None,
    *,
    sigmas: Sequence[float],
    alpha: float,
    beta: float,
    gamma: float | None,
    threshold: float | None,
    device: int,
    white_ridges: bool = False,
    mode: str = "nearest",
    save_outputs: bool = True,
    output_dir: Path | None = None,
    output_basename: str | None = None,
) -> float | None:
    """Load NIfTI from disk then call run_frangi_array."""
    img = nib.load(str(input_path))
    data = img.get_fdata(dtype=np.float32)
    spacing = img.header.get_zooms()[: data.ndim]

    label_data = None
    if label_path is not None and label_path.exists():
        lbl_img = nib.load(str(label_path))
        label_data = lbl_img.get_fdata()

    base_name = output_basename or f"frangi_{input_path.name}"
    return run_frangi_array(
        data,
        spacing,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        threshold=threshold,
        label_data=label_data,
        device=device,
        white_ridges=white_ridges,
        mode=mode,
        save_outputs=save_outputs,
        output_dir=output_dir,
        output_basename=base_name,
        affine=img.affine,
        header=img.header,
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Apply cucim Frangi filter (GPU) to a NIfTI volume."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input NIfTI.")
    parser.add_argument("--label", type=Path, default=None, help="Optional label NIfTI for Dice computation.")
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
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Write vesselness/label NIfTI outputs (default: no writes).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (required if --save_outputs).",
    )
    parser.add_argument(
        "--best_file",
        type=Path,
        default=None,
        help="Where to store/update best params JSON (default: OUTPUT_DIR/best.json or ./frangi_gpu_best.json).",
    )

    args = parser.parse_args(argv)
    sigmas = parse_sigmas(args.sigmas)
    save_outputs = bool(args.save_outputs)
    if save_outputs and args.output_dir is None:
        raise ValueError("--output_dir is required when --save_outputs is set.")

    best_file = args.best_file
    if best_file is None:
        if args.output_dir is not None:
            best_file = args.output_dir / "best.json"
        else:
            best_file = Path("frangi_gpu_best.json")

    dice_value = run_frangi_file(
        args.input,
        args.label,
        sigmas=sigmas,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        threshold=args.threshold,
        device=args.device,
        white_ridges=args.white_ridges,
        mode=args.mode,
        save_outputs=save_outputs,
        output_dir=args.output_dir,
    )

    if dice_value is not None:
        params_record = {
            "input": str(args.input),
            "label": str(args.label) if args.label is not None else None,
            "sigmas": sigmas,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "threshold": args.threshold,
            "white_ridges": args.white_ridges,
            "mode": args.mode,
            "device": args.device,
        }
        updated, prev_best = update_best_params(best_file, params_record, dice_value)
        if updated:
            note = " (first entry)" if prev_best is None else ""
            print(f"Saved best Dice={dice_value:.4f} to {best_file}{note}")
        elif prev_best is not None:
            print(f"Best remains {prev_best:.4f}; file unchanged at {best_file}")
    else:
        print("Dice not computed (need --label and --threshold); best params not updated.")

    return dice_value

if __name__ == "__main__":
    main()
