"""Bayesian optimization driver for Frangi parameters using preprocessed ROIs.

This script:
  - Loads images/labels directly (defaults: data/images, data/labels).
  - Evaluates Frangi GPU (frangi_gpu.py) with a given theta = (alpha, beta, gamma, threshold).
  - Uses a simple GP + Expected Improvement loop to search for the best mean Dice on train IDs.

Assumptions:
  - ROI images: data/images/image_<ID>.nii.gz (configurable via --roi_images_dir)
  - ROI labels: data/labels/label_<ID>.nii.gz (configurable via --roi_labels_dir)
  - Train IDs: data/splits/train_list.txt

Example:
export CUDA_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -m model_script.bo.frangi_BO \
  --train_list /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/splits/train_list.txt \
  --roi_images_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/images \
  --roi_labels_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/labels \
  --work_dir ./results/output_frangi_bo_seed0 \
  --sigmas 0.6,0.8,1.2,1.6,2.0 \
  --bounds_alpha 0.45,0.55 \
  --bounds_beta  0.45,0.55 \
  --bounds_gamma 3.0,8.0 \
  --bounds_thr   0.05,0.20 \
  --n_init 12 --n_iter 40 --batch_size 3 \
  --device 0  --seed 0 --theta_workers 3\
  --no_preload


  python -m model_script.bo.frangi_BO \
    --roi_images_dir ./data/images \
    --roi_labels_dir ./data/labels \
    --train_list data/splits/train_list.txt \
    --sigmas 0.6,0.8,1.2 \
    --work_dir results/output_frangi_bo \
    --device 0 \
    --n_init 12 --n_iter 80 --batch_size 64 --n_candidates 4000

Adjust bounds via --bounds_alpha 0.3,0.7, --bounds_beta ..., 
--bounds_gamma 10,25, --bounds_thr 0.05,0.4 as needed; 
--n_cases limits the train subset for faster loops.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in (None, ""):
    # Allow running as a script: add repo root so model_script is importable.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from model_script.bo import frangi_gpu  # normal packaged import
except ImportError:
    # Fallback if running from a copied script alongside frangi_gpu.py
    sys.path.append(str(Path(__file__).resolve().parent))
    import frangi_gpu

import nibabel as nib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator  # NEW
import tempfile # NEW

@dataclass
class Theta:
    alpha: float
    beta: float
    gamma: float
    thr: float

    def to_list(self) -> List[float]:
        return [self.alpha, self.beta, self.gamma, self.thr]


def parse_float_list(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


@dataclass
class RoiData:
    case_id: str
    image: np.ndarray
    label: np.ndarray | None
    affine: np.ndarray
    header: nib.Nifti1Header
    spacing: Tuple[float, ...]
    image_name: str


def load_ids(path: Path, max_cases: int | None) -> List[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if max_cases is not None and max_cases > 0:
        ids = ids[:max_cases]
    return ids


def normalize_id(case_id: str) -> str:
    """Strip leading prefixes like 'image_' so split files can use either style."""
    if case_id.startswith("image_"):
        return case_id[len("image_") :]
    return case_id


def sample_theta(rng: random.Random, bounds: dict[str, Tuple[float, float]]) -> Theta:
    return Theta(
        alpha=rng.uniform(*bounds["alpha"]),
        beta=rng.uniform(*bounds["beta"]),
        gamma=rng.uniform(*bounds["gamma"]),
        thr=rng.uniform(*bounds["thr"]),
    )


def preload_rois(
    train_ids: Sequence[str],
    roi_images_dir: Path,
    roi_labels_dir: Path,
) -> Dict[str, RoiData]:
    """Load ROI image/label pairs into memory once to avoid per-eval I/O."""
    cache: Dict[str, RoiData] = {}
    for case_id in tqdm(train_ids, desc="Preload ROIs", leave=False):
        img_path = roi_images_dir / f"image_{case_id}.nii.gz"
        lbl_path = roi_labels_dir / f"label_{case_id}.nii.gz"
        img = nib.load(str(img_path))
        data = img.get_fdata(dtype=np.float32)
        spacing = tuple(float(s) for s in img.header.get_zooms()[: data.ndim])

        label_data = None
        if lbl_path.exists():
            lbl_img = nib.load(str(lbl_path))
            label_data = np.asarray(lbl_img.get_fdata(), dtype=np.uint8)

        cache[case_id] = RoiData(
            case_id=case_id,
            image=data,
            label=label_data,
            affine=img.affine,
            header=img.header.copy(),
            spacing=spacing,
            image_name=img_path.name,
        )
    return cache


def run_frangi_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    sigmas: Sequence[float],
    theta: Theta,
    output_dir: Path | None,
    device: int,
    save_outputs: bool,
    roi_data: RoiData | None = None,
) -> float | None:
    """Run Frangi for one case and return Dice."""
    output_basename = f"frangi_{image_path.name}"
    # NEW code add here
    # --- NEW: TotalSegmentator lung+heart masking ---
    img = nib.load(str(image_path))
    orig_img = img.get_fdata(dtype=np.float32)

    # Run TotalSegmentator to get heart + lung
    with tempfile.TemporaryDirectory() as tmpdir:
        seg = totalsegmentator(
            input_path=str(image_path),
            output_path=tmpdir,
            task="total",
            quiet=True
        )
        # seg is dict of numpy arrays; combine heart + lungs
        mask = (seg["heart"] > 0) | (seg["lung_upper_lobe_left"] > 0) | (seg["lung_upper_lobe_right"] > 0) | \
               (seg["lung_lower_lobe_left"] > 0) | (seg["lung_lower_lobe_right"] > 0)

    masked_img = orig_img * mask.astype(np.float32)
    # overwrite roi_data image
    if roi_data is not None:
        roi_data.image = masked_img
    # -------------------------------------------------

    if roi_data is not None:
        return frangi_gpu.run_frangi_array(
            image=roi_data.image,
            spacing=roi_data.spacing,
            sigmas=sigmas,
            alpha=theta.alpha,
            beta=theta.beta,
            gamma=theta.gamma,
            threshold=theta.thr,
            label_data=roi_data.label,
            device=device,
            save_outputs=save_outputs,
            output_dir=output_dir,
            output_basename=output_basename,
            affine=roi_data.affine,
            header=roi_data.header,
        )

    return frangi_gpu.run_frangi_file(
        input_path=image_path,
        label_path=label_path,
        sigmas=sigmas,
        alpha=theta.alpha,
        beta=theta.beta,
        gamma=theta.gamma,
        threshold=theta.thr,
        device=device,
        save_outputs=save_outputs,
        output_dir=output_dir,
        output_basename=output_basename,
    )


def evaluate_theta(
    theta: Theta,
    sigmas: Sequence[float],
    train_ids: Sequence[str],
    roi_images_dir: Path,
    roi_labels_dir: Path,
    work_dir: Path,
    device: int,
    save_outputs: bool,
    roi_cache: Dict[str, RoiData] | None = None,
) -> float:
    """Compute mean Dice over train IDs for a theta."""
    dices: List[float] = []
    for case_id in tqdm(train_ids, desc="Eval theta", leave=False):
        img_path = roi_images_dir / f"image_{case_id}.nii.gz"
        lbl_path = roi_labels_dir / f"label_{case_id}.nii.gz"
        roi_entry = roi_cache.get(case_id) if roi_cache is not None else None
        if roi_entry is None and (not img_path.exists() or not lbl_path.exists()):
            raise FileNotFoundError(
                f"Missing ROI files for {case_id}. "
                f"Expected image at {img_path} and label at {lbl_path}. "
                "Check --roi_images_dir/--roi_labels_dir and filename pattern."
            )
        run_dir = work_dir / f"eval_{case_id}" if save_outputs else None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
        dice = run_frangi_case(
            case_id=case_id,
            image_path=img_path,
            label_path=lbl_path,
            sigmas=sigmas,
            theta=theta,
            output_dir=run_dir,
            device=device,
            save_outputs=save_outputs,
            roi_data=roi_entry,
        )
        if dice is not None:
            dices.append(dice)
    if not dices:
        return 0.0
    return float(np.mean(dices))


def persist_best_json(
    work_dir: Path, X_all: Sequence[Sequence[float]], y_all: Sequence[float]
) -> tuple[float, List[float]]:
    """Write best.json with the highest Dice seen so far.

    Returns the best Dice and corresponding theta list.
    """
    if not y_all:
        raise ValueError("persist_best_json called with no evaluations.")

    best_idx = int(np.argmax(y_all))
    best_theta = list(X_all[best_idx])
    best_score = float(y_all[best_idx])
    with (work_dir / "best.json").open("w") as f:
        json.dump({"theta": best_theta, "dice": best_score}, f, indent=2)

    return best_score, best_theta


def load_existing_history(
    work_dir: Path, n_init: int, batch_size: int
) -> tuple[List[dict], List[List[float]], List[float], int, int, int]:
    """Load prior evaluations from history.json if it exists.

    Returns (history, X_all, y_all, init_done, bo_iters_done, partial_batch).
    """
    history_path = work_dir / "history.json"
    if not history_path.exists():
        return [], [], [], 0, 0, 0

    try:
        raw = json.loads(history_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive parse
        print(f"Could not read {history_path}: {exc}; starting fresh history.")
        return [], [], [], 0, 0, 0

    history: List[dict] = []
    X_all: List[List[float]] = []
    y_all: List[float] = []
    for entry in raw:
        theta_dict = entry.get("theta", {}) if isinstance(entry, dict) else {}
        dice = entry.get("dice") if isinstance(entry, dict) else None
        try:
            theta_obj = Theta(
                alpha=float(theta_dict["alpha"]),
                beta=float(theta_dict["beta"]),
                gamma=float(theta_dict["gamma"]),
                thr=float(theta_dict["thr"]),
            )
            score = float(dice)
        except Exception:
            continue
        history.append({"theta": asdict(theta_obj), "dice": score})
        X_all.append(theta_obj.to_list())
        y_all.append(score)

    init_done = min(len(history), n_init)
    bo_evals_done = max(0, len(history) - init_done)
    bo_iters_done = bo_evals_done // batch_size
    partial_batch = bo_evals_done % batch_size

    if history:
        print(f"Loaded {len(history)} prior evals from {history_path}")
        if partial_batch:
            print(
                f"Warning: {partial_batch} eval(s) from an incomplete BO batch are present; "
                "continuing with new proposals."
            )

    return history, X_all, y_all, init_done, bo_iters_done, partial_batch


def expected_improvement(
    X: np.ndarray, model: GaussianProcessRegressor, y_best: float, xi: float = 0.01
) -> np.ndarray:
    """Compute EI for candidate points X (n, d)."""
    mu, sigma = model.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    Z = imp / sigma
    from scipy.stats import norm

    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei


def propose_next_batch(
    model: GaussianProcessRegressor,
    bounds: dict[str, Tuple[float, float]],
    n_candidates: int,
    batch_size: int,
    rng: np.random.Generator,
) -> List[Theta]:
    keys = ["alpha", "beta", "gamma", "thr"]
    lows = np.array([bounds[k][0] for k in keys])
    highs = np.array([bounds[k][1] for k in keys])
    X_rand = rng.uniform(lows, highs, size=(n_candidates, len(keys)))
    y_best = model.y_train_.max()
    ei = expected_improvement(X_rand, model, y_best=y_best)
    top_idx = np.argsort(ei)[::-1][:batch_size]
    thetas: List[Theta] = []
    for idx in top_idx:
        vals = X_rand[int(idx)]
        thetas.append(Theta(alpha=vals[0], beta=vals[1], gamma=vals[2], thr=vals[3]))
    return thetas


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    kernel = Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def _evaluate_theta_task(
    theta: Theta,
    args_tuple: Tuple[
        Sequence[float],
        Sequence[str],
        Path,
        Path,
        Path,
        int,
        bool,
    ],
) -> Tuple[Theta, float]:
    """Wrapper for multiprocessing; ROI cache is not used in parallel path."""
    sigmas, train_ids, roi_images_dir, roi_labels_dir, work_dir, device, save_outputs = args_tuple
    score = evaluate_theta(
        theta,
        sigmas=sigmas,
        train_ids=train_ids,
        roi_images_dir=roi_images_dir,
        roi_labels_dir=roi_labels_dir,
        work_dir=work_dir,
        device=device,
        save_outputs=save_outputs,
        roi_cache=None,
    )
    return theta, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization of Frangi params on preprocessed ROIs.")
    parser.add_argument("--train_list", type=Path, default=Path("data/splits/train_list.txt"))
    parser.add_argument("--roi_images_dir", type=Path, default=Path("data/images"))
    parser.add_argument("--roi_labels_dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--sigmas", type=str, default="0.6,0.8,1.2", help="Sigma ladder in mm.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--work_dir", type=Path, default=Path("results/output_frangi_bo"))
    parser.add_argument("--n_init", type=int, default=8)
    parser.add_argument("--n_iter", type=int, default=20, help="Additional BO iterations after init.")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of thetas to propose per BO iteration.")
    parser.add_argument("--n_candidates", type=int, default=1000, help="Random candidates to sample for EI maximization.")
    parser.add_argument(
        "--theta_workers",
        type=int,
        default=1,
        help="Parallel workers for evaluating theta batch (same GPU). Use >1 only if GPU memory allows.",
    )
    parser.add_argument("--n_cases", type=int, default=0, help="Subset of train cases to use (0=all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep_outputs",
        "--save_outputs",
        dest="save_outputs",
        action="store_true",
        help="Write Frangi vesselness/label NIfTI files for each eval (default: skip saving).",
    )
    parser.add_argument(
        "--no_preload",
        action="store_true",
        help="Disable ROI preloading; fall back to reading each ROI from disk per eval.",
    )
    parser.add_argument("--bounds_alpha", type=str, default="0.3,0.7")
    parser.add_argument("--bounds_beta", type=str, default="0.3,0.7")
    parser.add_argument("--bounds_gamma", type=str, default="10,25")
    parser.add_argument("--bounds_thr", type=str, default="0.05,0.4")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    sigmas = parse_float_list(args.sigmas)
    raw_train_ids = load_ids(args.train_list, args.n_cases if args.n_cases > 0 else None)
    if not raw_train_ids:
        print("No train IDs found.")
        return

    # Normalize IDs (strip 'image_' if present) and keep only those with ROI files.
    train_ids = []
    missing = []
    for cid_raw in raw_train_ids:
        cid = normalize_id(cid_raw)
        img_path = args.roi_images_dir / f"image_{cid}.nii.gz"
        lbl_path = args.roi_labels_dir / f"label_{cid}.nii.gz"
        if img_path.exists() and lbl_path.exists():
            train_ids.append(cid)
        else:
            missing.append(cid_raw)
    if missing:
        print(f"Skipping {len(missing)} IDs with missing ROI files (first few: {missing[:5]})")
    if not train_ids:
        print("No usable train IDs after checking ROI directories.")
        return

    roi_cache = None
    if not args.no_preload:
        if args.theta_workers > 1:
            print("Parallel theta evaluation requested; skipping ROI preloading to avoid large cross-process copies.")
        else:
            roi_cache = preload_rois(train_ids, args.roi_images_dir, args.roi_labels_dir)

    bounds = {
        "alpha": tuple(parse_float_list(args.bounds_alpha)),
        "beta": tuple(parse_float_list(args.bounds_beta)),
        "gamma": tuple(parse_float_list(args.bounds_gamma)),
        "thr": tuple(parse_float_list(args.bounds_thr)),
    }

    (
        history,
        X_all,
        y_all,
        init_done_loaded,
        bo_iters_done_loaded,
        partial_batch_loaded,
    ) = load_existing_history(work_dir, args.n_init, args.batch_size)

    best_score = -float("inf")
    best_theta_list: List[float] | None = None
    if y_all:
        best_score, best_theta_list = persist_best_json(work_dir, X_all, y_all)

    # Initial random design (skip those already completed)
    for i in range(init_done_loaded, args.n_init):
        theta = sample_theta(rng, bounds)
        score = evaluate_theta(
            theta,
            sigmas=sigmas,
            train_ids=train_ids,
            roi_images_dir=args.roi_images_dir,
            roi_labels_dir=args.roi_labels_dir,
            work_dir=work_dir,
            device=args.device,
            save_outputs=args.save_outputs,
            roi_cache=roi_cache,
        )
        X_all.append(theta.to_list())
        y_all.append(score)
        history.append({"theta": asdict(theta), "dice": score})
        print(f"[init {i+1}/{args.n_init}] Dice={score:.4f} for {theta}")

        prev_best = best_score
        best_score, best_theta_list = persist_best_json(work_dir, X_all, y_all)
        if best_score > prev_best + 1e-9:
            theta_named = Theta(*best_theta_list)
            print(f"  New best so far: Dice={best_score:.4f} at theta={theta_named}")

    total_evals = len(history)
    init_done = min(total_evals, args.n_init)
    bo_evals_done = max(0, total_evals - args.n_init)
    bo_iters_done = bo_evals_done // args.batch_size
    partial_batch = bo_evals_done % args.batch_size
    if partial_batch:
        print(
            f"Resuming with {partial_batch} eval(s) from an incomplete BO batch; new batch will start fresh."
        )
    if bo_iters_done or init_done:
        print(
            f"Resuming from {total_evals} evals done ({init_done} init, {bo_iters_done} full BO iterations)."
        )

    if bo_iters_done >= args.n_iter:
        print("All requested BO iterations already completed; skipping new proposals.")

    # BO loop
    for t in range(bo_iters_done, args.n_iter):
        X = np.array(X_all, dtype=float)
        y = np.array(y_all, dtype=float)
        gp = fit_gp(X, y)
        theta_batch = propose_next_batch(
            gp,
            bounds,
            n_candidates=args.n_candidates,
            batch_size=args.batch_size,
            rng=np_rng,
        )

        if args.theta_workers > 1:
            shared_args = (
                sigmas,
                train_ids,
                args.roi_images_dir,
                args.roi_labels_dir,
                work_dir,
                args.device,
                args.save_outputs,
            )
            spawn_ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.theta_workers, mp_context=spawn_ctx
            ) as ex:
                future_to_theta = {
                    ex.submit(_evaluate_theta_task, theta_next, shared_args): theta_next
                    for theta_next in theta_batch
                }
                completed = 0
                total = len(theta_batch)
                for future in concurrent.futures.as_completed(future_to_theta):
                    theta_next, score = future.result()
                    X_all.append(theta_next.to_list())
                    y_all.append(score)
                    history.append({"theta": asdict(theta_next), "dice": score})
                    completed += 1
                    print(
                        f"[iter {t+1}/{args.n_iter} batch {completed}/{total}] "
                        f"Dice={score:.4f} for {theta_next}"
                    )
                    prev_best = best_score
                    best_score, best_theta_list = persist_best_json(work_dir, X_all, y_all)
                    if best_score > prev_best + 1e-9:
                        theta_named = Theta(*best_theta_list)
                        print(
                            f"  New best so far: Dice={best_score:.4f} at theta={theta_named}"
                        )
        else:
            for b_idx, theta_next in enumerate(theta_batch, start=1):
                score = evaluate_theta(
                    theta_next,
                    sigmas=sigmas,
                    train_ids=train_ids,
                    roi_images_dir=args.roi_images_dir,
                    roi_labels_dir=args.roi_labels_dir,
                    work_dir=work_dir,
                    device=args.device,
                    save_outputs=args.save_outputs,
                    roi_cache=roi_cache,
                )
                X_all.append(theta_next.to_list())
                y_all.append(score)
                history.append({"theta": asdict(theta_next), "dice": score})
                print(f"[iter {t+1}/{args.n_iter} batch {b_idx}/{len(theta_batch)}] Dice={score:.4f} for {theta_next}")
                prev_best = best_score
                best_score, best_theta_list = persist_best_json(work_dir, X_all, y_all)
                if best_score > prev_best + 1e-9:
                    theta_named = Theta(*best_theta_list)
                    print(f"  New best so far: Dice={best_score:.4f} at theta={theta_named}")

        # Save incremental history
        with (work_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    if not y_all:
        print("No evaluations were run; nothing to report.")
        return

    best_score, best_theta_list = persist_best_json(work_dir, X_all, y_all)
    theta_named = Theta(*best_theta_list)
    print(f"Best Dice={best_score:.4f} at theta={theta_named}")


if __name__ == "__main__":
    main()
