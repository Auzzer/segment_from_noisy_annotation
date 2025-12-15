"""Bayesian optimization driver for Frangi parameters using preprocessed ROIs.

This script:
  - Loads preprocessed ROI images/labels (normalized, 1mm iso) from data_preproc.
  - Evaluates Frangi GPU (frangi_gpu.py) with a given theta = (alpha, beta, gamma, threshold).
  - Uses a simple GP + Expected Improvement loop to search for the best mean Dice on train IDs.

Assumptions:
  - ROI images: data_preproc/images_roi/image_<ID>.nii.gz
  - ROI labels: data_preproc/labels_roi/label_<ID>.nii.gz
  - Train IDs: data/splits/train_list.txt

Example:
  python frangi_BO.py \
    --train_list data/splits/train_list.txt \
    --roi_images_dir ./data_preproc/images_roi \
    --roi_labels_dir ./data_preproc/labels_roi \
    --work_dir results/output_frangi_bo \
    --sigmas 0.6,0.8,1.2 \
    --n_init 8 --n_iter 20
python frangi_BO.py \
  --roi_images_dir ./data_preproc/images_roi \
  --roi_labels_dir ./data_preproc/labels_roi \
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
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from tqdm import tqdm


DICE_RE = re.compile(r"Dice.*:\s*([0-9.]+)")


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


def run_frangi_case(
    image_path: Path,
    label_path: Path,
    sigmas: Sequence[float],
    theta: Theta,
    output_dir: Path,
    device: int,
) -> float | None:
    """Run frangi_gpu.py (in-process) for one case and return Dice."""
    import frangi_gpu

    argv = [
        "--input",
        str(image_path),
        "--label",
        str(label_path),
        "--output_dir",
        str(output_dir),
        "--sigmas",
        ",".join(str(s) for s in sigmas),
        "--alpha",
        str(theta.alpha),
        "--beta",
        str(theta.beta),
        "--gamma",
        str(theta.gamma),
        "--threshold",
        str(theta.thr),
        "--device",
        str(device),
    ]
    return frangi_gpu.main(argv)


def evaluate_theta(
    theta: Theta,
    sigmas: Sequence[float],
    train_ids: Sequence[str],
    roi_images_dir: Path,
    roi_labels_dir: Path,
    work_dir: Path,
    device: int,
    keep_outputs: bool,
) -> float:
    """Compute mean Dice over train IDs for a theta."""
    dices: List[float] = []
    for case_id in tqdm(train_ids, desc="Eval theta", leave=False):
        img_path = roi_images_dir / f"image_{case_id}.nii.gz"
        lbl_path = roi_labels_dir / f"label_{case_id}.nii.gz"
        if not img_path.exists() or not lbl_path.exists():
            raise FileNotFoundError(
                f"Missing ROI files for {case_id}. "
                f"Expected image at {img_path} and label at {lbl_path}. "
                "Check --roi_images_dir/--roi_labels_dir and filename pattern."
            )
        run_dir = work_dir / f"eval_{case_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        dice = run_frangi_case(
            image_path=img_path,
            label_path=lbl_path,
            sigmas=sigmas,
            theta=theta,
            output_dir=run_dir,
            device=device,
        )
        if dice is not None:
            dices.append(dice)
        if not keep_outputs:
            shutil.rmtree(run_dir, ignore_errors=True)
    if not dices:
        return 0.0
    return float(np.mean(dices))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization of Frangi params on preprocessed ROIs.")
    parser.add_argument("--train_list", type=Path, default=Path("data/splits/train_list.txt"))
    parser.add_argument("--roi_images_dir", type=Path, default=Path("data_preproc/images_roi"))
    parser.add_argument("--roi_labels_dir", type=Path, default=Path("data_preproc/labels_roi"))
    parser.add_argument("--sigmas", type=str, default="0.6,0.8,1.2", help="Sigma ladder in mm.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--work_dir", type=Path, default=Path("results/output_frangi_bo"))
    parser.add_argument("--n_init", type=int, default=8)
    parser.add_argument("--n_iter", type=int, default=20, help="Additional BO iterations after init.")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of thetas to propose per BO iteration.")
    parser.add_argument("--n_candidates", type=int, default=1000, help="Random candidates to sample for EI maximization.")
    parser.add_argument("--n_cases", type=int, default=0, help="Subset of train cases to use (0=all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep_outputs", action="store_true", help="Keep intermediate frangi outputs.")
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

    bounds = {
        "alpha": tuple(parse_float_list(args.bounds_alpha)),
        "beta": tuple(parse_float_list(args.bounds_beta)),
        "gamma": tuple(parse_float_list(args.bounds_gamma)),
        "thr": tuple(parse_float_list(args.bounds_thr)),
    }

    history = []
    X_all: List[List[float]] = []
    y_all: List[float] = []

    # Initial random design
    for i in range(args.n_init):
        theta = sample_theta(rng, bounds)
        score = evaluate_theta(
            theta,
            sigmas=sigmas,
            train_ids=train_ids,
            roi_images_dir=args.roi_images_dir,
            roi_labels_dir=args.roi_labels_dir,
            work_dir=work_dir,
            device=args.device,
            keep_outputs=args.keep_outputs,
        )
        X_all.append(theta.to_list())
        y_all.append(score)
        history.append({"theta": asdict(theta), "dice": score})
        print(f"[init {i+1}/{args.n_init}] Dice={score:.4f} for {theta}")

    # BO loop
    for t in range(args.n_iter):
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
        for b_idx, theta_next in enumerate(theta_batch, start=1):
            score = evaluate_theta(
                theta_next,
                sigmas=sigmas,
                train_ids=train_ids,
                roi_images_dir=args.roi_images_dir,
                roi_labels_dir=args.roi_labels_dir,
                work_dir=work_dir,
                device=args.device,
                keep_outputs=args.keep_outputs,
            )
            X_all.append(theta_next.to_list())
            y_all.append(score)
            history.append({"theta": asdict(theta_next), "dice": score})
            print(f"[iter {t+1}/{args.n_iter} batch {b_idx}/{len(theta_batch)}] Dice={score:.4f} for {theta_next}")

        # Save incremental history
        with (work_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    best_idx = int(np.argmax(y_all))
    best_theta = X_all[best_idx]
    best_score = y_all[best_idx]
    print(f"Best Dice={best_score:.4f} at theta={best_theta}")
    with (work_dir / "best.json").open("w") as f:
        json.dump({"theta": best_theta, "dice": best_score}, f, indent=2)


if __name__ == "__main__":
    main()
