"""
python model_script/train/mtcl_3level.py \
--data_dir ./data --output_dir ./results/output_mtcl_3level 

Mean-Teacher + Confident Learning (three-source) trainer for binary vessels.

This script fine-tunes a 10-model U-Net ensemble with HQ labels, MQ offline
predictions, and VQ Frangi priors while maintaining EMA teachers. It implements
the MTCL blueprint provided in the request with ensemble-to-ensemble
consistency, forward-correction for noisy sources, and periodic CL refreshes.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import _normalize_image_id  
from utils.losses import calculate_dice_score  
from utils.unet_model import UNet3D  


# ------------------------------- I/O helpers ------------------------------- #
def load_nifti_array(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def find_matching_file(root: Path, base_id: str) -> Path | None:
    suffix = base_id.split("_", 1)[1] if "_" in base_id else base_id
    candidates = [
        root / f"{base_id}.nii.gz",
        root / f"{base_id}.nii",
        root / f"label_{suffix}.nii.gz",
        root / f"image_{suffix}.nii.gz",
        root / f"pred_{suffix}.nii.gz",
        root / f"pred_{base_id}.nii.gz",
        root / f"frangi_{base_id}.nii.gz",
        root / f"frangi_{base_id}_label.nii.gz",
        root / f"frangi_{suffix}_label.nii.gz",
    ]
    for cand in candidates:
        if cand is not None and cand.exists():
            return cand
    return None


def soften_mask(prob_like: np.ndarray, epsilon: float) -> np.ndarray:
    """Convert mask/logit to 2ch probabilities."""
    arr = prob_like.astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 2:
        probs = arr
    elif arr.ndim == 4 and arr.shape[0] == 1:
        p1 = torch.sigmoid(torch.from_numpy(arr[0])).numpy()
        probs = np.stack([1.0 - p1, p1], axis=0)
    else:
        p1 = arr
        if p1.max() > 1.0 or p1.min() < 0.0:
            p1 = 1.0 / (1.0 + np.exp(-p1))
        if set(np.unique(p1).tolist()) <= {0.0, 1.0}:
            p1 = p1 * (1.0 - epsilon) + (1.0 - p1) * epsilon
        probs = np.stack([1.0 - p1, p1], axis=0)
    probs = np.clip(probs, 1e-5, 1.0 - 1e-5)
    return probs


def apply_temperature_np(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=0, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=0, keepdims=True)


def to_one_hot(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    labels = labels.squeeze(1)  # [B, D, H, W]
    oh = F.one_hot(labels.long(), num_classes=num_classes)  # [B, D, H, W, K]
    return oh.permute(0, 4, 1, 2, 3).float()  # [B, K, D, H, W]


# ------------------------------- Dataset ----------------------------------- #
@dataclass
class SourceWeighting:
    gamma_mq: float
    gamma_vq: float
    kappa_v: float
    epsilon_mask: float


class MultiSourceDataset(Dataset):
    """
    Loads HQ labels, MQ offline predictions, and VQ Frangi priors. Returns
    aligned patches/volumes with optional soft labels refreshed by CL.
    """

    def __init__(
        self,
        data_root: Path,
        ids: Iterable[str],
        weighting: SourceWeighting,
        *,
        patch_size: Tuple[int, int, int] | None = (64, 64, 64),
        use_soft_labels: bool = True,
        temperature_offline: float = 1.0,
        mq_subdir: str = "",
    ) -> None:
        self.data_root = Path(data_root)
        self.ids = [_normalize_image_id(i) for i in ids]
        self.patch_size = patch_size
        self.temperature_offline = temperature_offline
        self.weighting = weighting
        self.use_soft_labels = use_soft_labels
        self.mq_subdir = mq_subdir
        self._missing_mq_notified: set[str] = set()

        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.mq_dir = self.data_root / "unet_prediction" / self.mq_subdir if self.mq_subdir else self.data_root / "unet_prediction"
        self.vq_dir = self.data_root / "frangi"
        self.mq_soft_dir = self.data_root / "mq_soft"
        self.vq_soft_dir = self.data_root / "vq_soft"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        case_id = self.ids[idx]
        image_path = find_matching_file(self.images_dir, case_id)
        label_path = find_matching_file(self.labels_dir, case_id)
        if image_path is None or label_path is None:
            raise FileNotFoundError(f"Missing image/label for id {case_id}")

        image = load_nifti_array(image_path)
        label = load_nifti_array(label_path)

        if self.use_soft_labels:
            mq_soft_path = find_matching_file(self.mq_soft_dir, case_id)
            vq_soft_path = find_matching_file(self.vq_soft_dir, case_id)
        else:
            mq_soft_path = vq_soft_path = None

        if mq_soft_path and mq_soft_path.exists():
            mq_probs = soften_mask(load_nifti_array(mq_soft_path), self.weighting.epsilon_mask)
            mq_missing = False
        else:
            mq_missing = False
            base_mq_path = find_matching_file(self.mq_dir, case_id)
            if base_mq_path is None:
                mq_missing = True
                if case_id not in self._missing_mq_notified:
                    print(f"[Warning] Missing MQ prediction for id {case_id}; using zero-weight MQ targets.")
                    self._missing_mq_notified.add(case_id)
                mq_probs = np.stack(
                    [np.ones_like(label, dtype=np.float32), np.zeros_like(label, dtype=np.float32)],
                    axis=0,
                )
            else:
                mq_probs = soften_mask(load_nifti_array(base_mq_path), self.weighting.epsilon_mask)
                mq_probs = apply_temperature_np(mq_probs, self.temperature_offline)

        if vq_soft_path and vq_soft_path.exists():
            vq_probs = soften_mask(load_nifti_array(vq_soft_path), self.weighting.epsilon_mask)
            v_prior = vq_probs[1]
        else:
            v_path = find_matching_file(self.vq_dir, case_id)
            if v_path is None:
                raise FileNotFoundError(f"Missing VQ prior for id {case_id}")
            v_prior = load_nifti_array(v_path)
            v_prior = np.clip(v_prior, 0.0, 1.0)
            vq_probs = np.stack([1.0 - v_prior, v_prior], axis=0)

        # Extract patch (shared crop for all arrays)
        image, label, mq_probs, vq_probs = self._maybe_crop(
            image, label, mq_probs, vq_probs
        )

        # Normalize image
        image = (image - image.mean()) / (image.std() + 1e-6)

        # Build weights
        mq_conf = mq_probs.max(axis=0)
        mq_weight = np.clip(self.weighting.gamma_mq * mq_conf, 0.0, 1.0)
        if mq_missing:
            mq_weight = np.zeros_like(mq_weight, dtype=np.float32)

        v_conf = vq_probs[1]
        v_gate = (v_conf >= self.weighting.kappa_v).astype(np.float32)
        v_weight = np.clip(self.weighting.gamma_vq * v_conf * v_gate, 0.0, 1.0)

        # Torch tensors
        image_t = torch.from_numpy(image[None, ...])  # (1,D,H,W)
        label_t = torch.from_numpy(label[None, ...])
        mq_t = torch.from_numpy(mq_probs)
        vq_t = torch.from_numpy(vq_probs)
        mq_w_t = torch.from_numpy(mq_weight[None, ...])
        vq_w_t = torch.from_numpy(v_weight[None, ...])

        return {
            "id": case_id,
            "image": image_t,
            "label": label_t,
            "mq_soft": mq_t,
            "vq_soft": vq_t,
            "mq_weight": mq_w_t,
            "vq_weight": vq_w_t,
        }

    def _maybe_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
        mq_probs: np.ndarray,
        vq_probs: np.ndarray,
    ):
        if self.patch_size is None:
            return image, label, mq_probs, vq_probs

        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        if pad_d or pad_h or pad_w:
            pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
            image = np.pad(image, pad_width, mode="constant")
            label = np.pad(label, pad_width, mode="constant")
            mq_probs = np.pad(mq_probs, ((0, 0), pad_width), mode="constant")
            vq_probs = np.pad(vq_probs, ((0, 0), pad_width), mode="constant")
            d, h, w = image.shape

        d_start = np.random.randint(0, d - pd + 1) if d > pd else 0
        h_start = np.random.randint(0, h - ph + 1) if h > ph else 0
        w_start = np.random.randint(0, w - pw + 1) if w > pw else 0

        slicer = np.s_[d_start:d_start + pd, h_start:h_start + ph, w_start:w_start + pw]
        return (
            image[slicer],
            label[slicer],
            mq_probs[(slice(None),) + slicer],
            vq_probs[(slice(None),) + slicer],
        )


# ------------------------------- Losses ------------------------------------ #
def weighted_ce(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    noise_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_probs = probs.permute(0, 2, 3, 4, 1).reshape(-1, 2)
    flat_target = target.permute(0, 2, 3, 4, 1).reshape(-1, 2)

    if noise_matrix is not None:
        corrected = torch.matmul(flat_probs, noise_matrix.to(flat_probs.device))
    else:
        corrected = flat_probs
    corrected = torch.clamp(corrected, 1e-6, 1.0 - 1e-6)
    log_p = torch.log(corrected)

    weights = class_weights.to(probs.device)
    ce = -(flat_target * log_p * weights).sum(dim=1)
    if weight_map is not None:
        voxel_w = weight_map.reshape(-1).to(probs.device)
        ce = ce * voxel_w
    return ce.mean()


def dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if weight_map is not None:
        weight_map = weight_map.to(probs.device)
        weight_map = weight_map.expand_as(probs)
        probs = probs * weight_map
        target = target * weight_map

    dims = tuple(range(2, probs.ndim))
    intersect = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims) + eps
    dice_per_class = 2.0 * intersect / denom
    weights = class_weights.to(probs.device)
    loss = (weights * (1.0 - dice_per_class)).sum() / weights.sum()
    return loss


# --------------------------- Training utilities --------------------------- #
def build_students(num_models: int, device: torch.device) -> List[UNet3D]:
    return [UNet3D(n_channels=1, n_classes=1, trilinear=True).to(device) for _ in range(num_models)]


def load_pretrained_students(students: List[UNet3D], checkpoint_root: Path, device: torch.device):
    for i, model in enumerate(students):
        ckpt_dir = Path(checkpoint_root) / f"model_{i}"
        candidates = [ckpt_dir / "best.pth", ckpt_dir / "latest.pth"]
        loaded = False
        for cand in candidates:
            if cand.exists():
                state = torch.load(cand, map_location=device)
                model.load_state_dict(state["model_state_dict"])
                loaded = True
                print(f"Loaded model_{i} from {cand}")
                break
        if not loaded:
            print(f"Warning: no checkpoint found for model_{i} under {ckpt_dir}; using fresh init.")


def make_ema_copies(students: List[UNet3D], device: torch.device) -> List[UNet3D]:
    teachers = []
    for student in students:
        t = UNet3D(n_channels=1, n_classes=1, trilinear=True).to(device)
        t.load_state_dict(student.state_dict())
        for p in t.parameters():
            p.requires_grad_(False)
        t.eval()
        teachers.append(t)
    return teachers


def update_ema(teacher: nn.Module, student: nn.Module, alpha: float):
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.mul_(alpha).add_(s_p.data, alpha=1.0 - alpha)


def student_teacher_views(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    noise_std = 0.05
    brightness = 0.1
    contrast = 0.1
    def _jitter(x: torch.Tensor) -> torch.Tensor:
        out = x
        if brightness > 0:
            delta = (torch.rand(1, device=x.device) - 0.5) * 2 * brightness
            out = out + delta
        if contrast > 0:
            scale = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 2 * contrast
            out = out * scale
        if noise_std > 0:
            out = out + noise_std * torch.randn_like(out)
        return out
    return _jitter(images), _jitter(images)


def stack_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p1 = torch.sigmoid(logits)
    p0 = 1.0 - p1
    return torch.cat([p0, p1], dim=1)


def estimate_class_weights(label_paths: List[Path], gamma: float = 0.5) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.float64)
    for path in label_paths:
        arr = load_nifti_array(path)
        counts[1] += (arr > 0.5).sum()
        counts[0] += (arr <= 0.5).sum()
    freqs = counts / counts.sum().clamp(min=1.0)
    weights = (freqs + 1e-6).pow(-gamma)
    weights = weights / weights.sum()
    return weights.float()


def prepare_id_lists(split_dir: Path, train_name: str, val_name: str) -> Tuple[List[str], List[str]]:
    def _read_list(path: Path) -> List[str]:
        with open(path, "r") as f:
            return [_normalize_image_id(line.strip()) for line in f if line.strip()]
    return _read_list(split_dir / train_name), _read_list(split_dir / val_name)


def assemble_label_paths(data_root: Path, ids: Iterable[str]) -> List[Path]:
    labels_dir = Path(data_root) / "labels"
    paths = []
    for cid in ids:
        p = find_matching_file(labels_dir, cid)
        if p:
            paths.append(p)
    return paths


def mean_confidence_mq(data_root: Path, ids: Iterable[str], epsilon: float, temp: float) -> float:
    mq_dir = Path(data_root) / "unet_prediction"
    vals = []
    for cid in ids:
        p = find_matching_file(mq_dir, cid)
        if p is None:
            continue
        probs = soften_mask(load_nifti_array(p), epsilon)
        probs = apply_temperature_np(probs, temp)
        vals.append(probs.max(axis=0).mean())
    return float(np.mean(vals)) if vals else 0.5


def mean_confidence_vq(data_root: Path, ids: Iterable[str], kappa_v: float) -> float:
    vq_dir = Path(data_root) / "frangi"
    vals = []
    for cid in ids:
        p = find_matching_file(vq_dir, cid)
        if p is None:
            continue
        v = np.clip(load_nifti_array(p), 0.0, 1.0)
        vals.append((v * (v >= kappa_v)).mean())
    return float(np.mean(vals)) if vals else 0.5


# --------------------------- Confident Learning --------------------------- #
@torch.no_grad()
def run_cl_pass(
    teachers: List[nn.Module],
    device: torch.device,
    dataset: MultiSourceDataset,
    *,
    out_soft_dir_mq: Path,
    out_soft_dir_vq: Path,
    out_noise_dir: Path,
    tau_percentile_vq: float,
    lambda_tau_mq: float,
    lambda_tau_vq: float,
    kappa_bg: float,
    temp_teacher: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    out_soft_dir_mq.mkdir(parents=True, exist_ok=True)
    out_soft_dir_vq.mkdir(parents=True, exist_ok=True)
    out_noise_dir.mkdir(parents=True, exist_ok=True)

    C_mq = torch.zeros((2, 2), device=device)
    C_vq = torch.zeros((2, 2), device=device)
    tau_mq = torch.zeros(2, device=device)
    tau_vq = torch.zeros(2, device=device)

    def teacher_ensemble(x: torch.Tensor) -> torch.Tensor:
        preds = []
        for t in teachers:
            t.eval()
            preds.append(stack_probs_from_logits(t(x)))
        return torch.stack(preds).mean(dim=0)

    # Pass 1: collect teacher probs + observed labels for thresholds
    teacher_probs_store: Dict[str, torch.Tensor] = {}
    mq_obs_store: Dict[str, torch.Tensor] = {}
    vq_obs_store: Dict[str, torch.Tensor] = {}
    vq_conf_store: Dict[str, torch.Tensor] = {}

    for batch in tqdm(loader, desc="CL pass (teacher forward)", leave=False):
        imgs = batch["image"].to(device)
        cid = batch["id"][0]
        tq = teacher_ensemble(imgs)
        if temp_teacher != 1.0:
            tq = torch.softmax(torch.log(torch.clamp(tq, 1e-6, 1.0 - 1e-6)) / temp_teacher, dim=1)
        teacher_probs_store[cid] = tq.squeeze(0).cpu()

        mq_soft = batch["mq_soft"].to(device)
        vq_soft = batch["vq_soft"].to(device)
        mq_labels = mq_soft.argmax(dim=1, keepdim=True)
        vq_labels = vq_soft.argmax(dim=1, keepdim=True)
        mq_obs_store[cid] = mq_labels.cpu()
        vq_obs_store[cid] = vq_labels.cpu()
        vq_conf_store[cid] = vq_soft[:, 1].squeeze(0).cpu()

    # Thresholds
    for cls in (0, 1):
        mq_vals = []
        vq_vals = []
        for cid in teacher_probs_store:
            tq = teacher_probs_store[cid]
            mq_lbl = mq_obs_store[cid]
            vq_lbl = vq_obs_store[cid]
            mq_mask = (mq_lbl.squeeze(0).squeeze(0) == cls)
            vq_mask = (vq_lbl.squeeze(0).squeeze(0) == cls)
            if mq_mask.any():
                mq_vals.append(tq[cls][mq_mask])
            if vq_mask.any():
                vq_conf = vq_conf_store[cid]
                gate = vq_conf >= dataset.weighting.kappa_v
                gated = vq_mask & gate
                if gated.any():
                    vq_vals.append(tq[cls][gated])
        if mq_vals:
            cat = torch.cat(mq_vals)
            tau_mq[cls] = cat.mean()
        else:
            tau_mq[cls] = 0.5
        if vq_vals:
            cat = torch.cat(vq_vals)
            tau_vq[cls] = torch.quantile(cat, tau_percentile_vq)
        else:
            tau_vq[cls] = 0.5

    # Pass 2: confident joint + relabel
    for batch in tqdm(loader, desc="CL pass (relabel)", leave=False):
        imgs = batch["image"].to(device)
        cid = batch["id"][0]
        tq = teacher_probs_store[cid].to(device)
        mq_soft = batch["mq_soft"].to(device).squeeze(0)
        vq_soft = batch["vq_soft"].to(device).squeeze(0)
        mq_weight = batch["mq_weight"].to(device).squeeze(0).squeeze(0)
        vq_weight = batch["vq_weight"].to(device).squeeze(0).squeeze(0)
        vq_conf = vq_conf_store[cid].to(device)

        mq_lbl = mq_soft.argmax(dim=0)
        vq_lbl = vq_soft.argmax(dim=0)
        top = tq.argmax(dim=0)
        conf = tq.max(dim=0).values

        is_conf_mq = torch.stack([(top == i) & (conf >= tau_mq[i]) for i in (0, 1)], dim=0)
        is_conf_vq = torch.stack([(top == i) & (conf >= tau_vq[i]) for i in (0, 1)], dim=0)

        gate_v = vq_weight > 0
        for i in (0, 1):
            for j in (0, 1):
                C_mq[i, j] += ((mq_lbl == j) & is_conf_mq[i]).sum()
                C_vq[i, j] += (((vq_lbl == j) & is_conf_vq[i] & gate_v)).sum()

        mq_soft_new = mq_soft.clone()
        vq_soft_new = vq_soft.clone()
        teacher_pos = int((top == 1).sum().item())
        min_cap = max(1, int(0.001 * top.numel()))
        limit_bg_to_vessel_mq = int(kappa_bg * max(teacher_pos, min_cap))
        limit_bg_to_vessel_vq = int(kappa_bg * max(teacher_pos, min_cap))

        for i in (0, 1):
            for j in (0, 1):
                if i == j:
                    continue
                mask_mq = (mq_lbl == j) & is_conf_mq[i]
                if i == 1 and j == 0:
                    if limit_bg_to_vessel_mq <= 0:
                        mask_mq = torch.zeros_like(mask_mq, dtype=torch.bool)
                    elif mask_mq.sum() > limit_bg_to_vessel_mq:
                        conf_vals = conf[mask_mq]
                        topk = torch.topk(conf_vals, k=limit_bg_to_vessel_mq).values.min()
                        mask_mq = mask_mq & (conf >= topk)
                mq_soft_new[:, mask_mq] = (
                    (1.0 - lambda_tau_mq) * F.one_hot(torch.tensor(j), num_classes=2).float().to(device)[:, None]
                    + lambda_tau_mq * F.one_hot(torch.tensor(i), num_classes=2).float().to(device)[:, None]
                )

                mask_vq = (vq_lbl == j) & is_conf_vq[i] & gate_v
                if i == 1 and j == 0:
                    if limit_bg_to_vessel_vq <= 0:
                        mask_vq = torch.zeros_like(mask_vq, dtype=torch.bool)
                    elif mask_vq.sum() > limit_bg_to_vessel_vq:
                        conf_vals = conf[mask_vq]
                        topk = torch.topk(conf_vals, k=limit_bg_to_vessel_vq).values.min()
                        mask_vq = mask_vq & (conf >= topk)
                vq_soft_new[:, mask_vq] = (
                    (1.0 - lambda_tau_vq) * F.one_hot(torch.tensor(j), num_classes=2).float().to(device)[:, None]
                    + lambda_tau_vq * tq[:, mask_vq]
                )

        mq_out = torch.clamp(mq_soft_new.cpu().numpy(), 1e-5, 1.0 - 1e-5)
        vq_out = torch.clamp(vq_soft_new.cpu().numpy(), 1e-5, 1.0 - 1e-5)

        mq_src = find_matching_file(dataset.mq_dir, cid)
        vq_src = find_matching_file(dataset.vq_dir, cid)
        mq_affine = nib.load(str(mq_src)).affine if mq_src and mq_src.exists() else np.eye(4)
        vq_affine = nib.load(str(vq_src)).affine if vq_src and vq_src.exists() else np.eye(4)
        mq_out_nifti = nib.Nifti1Image(mq_out.astype(np.float32), affine=mq_affine)
        vq_out_nifti = nib.Nifti1Image(vq_out.astype(np.float32), affine=vq_affine)
        nib.save(mq_out_nifti, out_soft_dir_mq / f"{cid}.nii.gz")
        nib.save(vq_out_nifti, out_soft_dir_vq / f"{cid}.nii.gz")

    eps = 1.0
    noise_mq = torch.zeros((2, 2), device=device)
    noise_vq = torch.zeros((2, 2), device=device)
    for i in (0, 1):
        noise_mq[i] = (C_mq[i] + eps) / (C_mq[i].sum() + 2 * eps)
        noise_vq[i] = (C_vq[i] + eps) / (C_vq[i].sum() + 2 * eps)

    np.save(out_noise_dir / "T_MQ.npy", noise_mq.cpu().numpy())
    np.save(out_noise_dir / "T_VQ.npy", noise_vq.cpu().numpy())
    return noise_mq, noise_vq


# ------------------------------- Training ---------------------------------- #
@dataclass
class MTCLArgs:
    data_dir: Path
    checkpoint_root: Path
    output_dir: Path
    train_split: str
    val_split: str
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    ema_alpha: float
    lambda_ce: float
    lambda_dice: float
    lambda_c_max: float
    lambda_mq_max: float
    lambda_vq_max: float
    gamma_class: float
    gamma_mq_target: float
    gamma_vq_target: float
    kappa_v: float
    kappa_bg: float
    lambda_tau_mq: float
    lambda_tau_vq: float
    tau_percentile_vq: float
    temp_offline: float
    temp_teacher: float
    epsilon_mask: float
    patch_size: Tuple[int, int, int] | None
    device: str
    cl_interval: int
    mq_subdir: str


def parse_args() -> MTCLArgs:
    p = argparse.ArgumentParser(description="Three-source MTCL trainer for vessels")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data"),
    )
    p.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_ensemble/checkpoints"),
    )
    p.add_argument("--output_dir", type=Path, default=Path("results/output_mtcl_3level"))
    p.add_argument("--train_split", type=str, default="train_list.txt")
    p.add_argument("--val_split", type=str, default="test_list.txt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_alpha", type=float, default=0.99)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_dice", type=float, default=1.0)
    p.add_argument("--lambda_c_max", type=float, default=0.5)
    p.add_argument("--lambda_mq_max", type=float, default=0.5)
    p.add_argument("--lambda_vq_max", type=float, default=0.2)
    p.add_argument("--gamma_class", type=float, default=0.5)
    p.add_argument("--gamma_mq_target", type=float, default=0.6)
    p.add_argument("--gamma_vq_target", type=float, default=0.15)
    p.add_argument("--kappa_v", type=float, default=0.7)
    p.add_argument("--kappa_bg", type=float, default=0.1)
    p.add_argument("--lambda_tau_mq", type=float, default=0.8)
    p.add_argument("--lambda_tau_vq", type=float, default=0.9)
    p.add_argument("--tau_percentile_vq", type=float, default=0.85)
    p.add_argument("--temp_offline", type=float, default=1.0)
    p.add_argument("--temp_teacher", type=float, default=1.0)
    p.add_argument("--epsilon_mask", type=float, default=0.05)
    p.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cl_interval", type=int, default=1)
    p.add_argument("--mq_subdir", type=str, default="ensemble_lungmasked")
    args = p.parse_args()
    return MTCLArgs(
        data_dir=args.data_dir,
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        ema_alpha=args.ema_alpha,
        lambda_ce=args.lambda_ce,
        lambda_dice=args.lambda_dice,
        lambda_c_max=args.lambda_c_max,
        lambda_mq_max=args.lambda_mq_max,
        lambda_vq_max=args.lambda_vq_max,
        gamma_class=args.gamma_class,
        gamma_mq_target=args.gamma_mq_target,
        gamma_vq_target=args.gamma_vq_target,
        kappa_v=args.kappa_v,
        kappa_bg=args.kappa_bg,
        lambda_tau_mq=args.lambda_tau_mq,
        lambda_tau_vq=args.lambda_tau_vq,
        tau_percentile_vq=args.tau_percentile_vq,
        temp_offline=args.temp_offline,
        temp_teacher=args.temp_teacher,
        epsilon_mask=args.epsilon_mask,
        patch_size=tuple(args.patch_size) if args.patch_size else None,
        device=args.device,
        cl_interval=args.cl_interval,
        mq_subdir=args.mq_subdir,
    )


def train_mtcl(args: MTCLArgs):
    req = args.device.lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    elif req.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    noise_dir = args.output_dir / "noise_mats"
    mq_soft_dir = args.data_dir / "mq_soft"
    vq_soft_dir = args.data_dir / "vq_soft"

    train_ids, val_ids = prepare_id_lists(args.data_dir / "splits", args.train_split, args.val_split)
    class_weights = estimate_class_weights(assemble_label_paths(args.data_dir, train_ids), gamma=args.gamma_class)

    mean_conf_mq = mean_confidence_mq(args.data_dir, train_ids, args.epsilon_mask, args.temp_offline)
    mean_conf_vq = mean_confidence_vq(args.data_dir, train_ids, args.kappa_v)
    gamma_mq = args.gamma_mq_target / max(mean_conf_mq, 1e-4)
    gamma_vq = args.gamma_vq_target / max(mean_conf_vq, 1e-4)
    weighting = SourceWeighting(
        gamma_mq=gamma_mq,
        gamma_vq=gamma_vq,
        kappa_v=args.kappa_v,
        epsilon_mask=args.epsilon_mask,
    )

    train_ds = MultiSourceDataset(
        args.data_dir,
        train_ids,
        weighting,
        patch_size=args.patch_size,
        temperature_offline=args.temp_offline,
        use_soft_labels=True,
        mq_subdir=args.mq_subdir,
    )
    val_ds = MultiSourceDataset(
        args.data_dir,
        val_ids,
        weighting,
        patch_size=args.patch_size,
        temperature_offline=args.temp_offline,
        use_soft_labels=True,
        mq_subdir=args.mq_subdir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    students = build_students(10, device)
    load_pretrained_students(students, args.checkpoint_root, device)
    teachers = make_ema_copies(students, device)
    opts = [torch.optim.Adam(m.parameters(), lr=args.lr) for m in students]

    noise_mq = torch.eye(2, device=device)
    noise_vq = torch.eye(2, device=device)

    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
            lambda_c = args.lambda_c_max * 0.5
            lambda_mq = 0.0
            lambda_vq = 0.0
        elif epoch == 2:
            lambda_c = args.lambda_c_max
            lambda_mq = args.lambda_mq_max * 0.5
            lambda_vq = 0.0
        elif epoch == 3:
            lambda_c = args.lambda_c_max
            lambda_mq = args.lambda_mq_max
            lambda_vq = args.lambda_vq_max * 0.5
        else:
            lambda_c = args.lambda_c_max
            lambda_mq = args.lambda_mq_max
            lambda_vq = args.lambda_vq_max

        students_loss, students_dice = run_train_epoch(
            students,
            teachers,
            train_loader,
            opts,
            device,
            class_weights,
            noise_mq,
            noise_vq,
            lambda_c=lambda_c,
            lambda_mq=lambda_mq,
            lambda_vq=lambda_vq,
            lambda_ce=args.lambda_ce,
            lambda_dice=args.lambda_dice,
            ema_alpha=args.ema_alpha,
        )
        val_loss, val_dice = evaluate_ensemble(students, val_loader, device)

        print(
            f"Epoch {epoch}: train_loss={students_loss:.4f}, train_dice={students_dice:.4f}, "
            f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "noise_mq": noise_mq.cpu().numpy(),
            "noise_vq": noise_vq.cpu().numpy(),
        }
        for i, m in enumerate(students):
            ckpt[f"model_{i}"] = m.state_dict()
        torch.save(ckpt, args.output_dir / "checkpoints" / "latest.pth")
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(ckpt, args.output_dir / "checkpoints" / "best.pth")

        if epoch >= 2 and ((epoch % args.cl_interval) == 0):
            cl_dataset = MultiSourceDataset(
                args.data_dir,
                train_ids,
                weighting,
                patch_size=args.patch_size,
                temperature_offline=args.temp_offline,
                use_soft_labels=True,
                mq_subdir=args.mq_subdir,
            )
            noise_mq, noise_vq = run_cl_pass(
                teachers,
                device,
                cl_dataset,
                out_soft_dir_mq=mq_soft_dir,
                out_soft_dir_vq=vq_soft_dir,
                out_noise_dir=noise_dir,
                tau_percentile_vq=args.tau_percentile_vq,
                lambda_tau_mq=args.lambda_tau_mq,
                lambda_tau_vq=args.lambda_tau_vq,
                kappa_bg=args.kappa_bg,
                temp_teacher=args.temp_teacher,
            )


def run_train_epoch(
    students: List[nn.Module],
    teachers: List[nn.Module],
    loader: DataLoader,
    opts: List[torch.optim.Optimizer],
    device: torch.device,
    class_weights: torch.Tensor,
    noise_mq: torch.Tensor,
    noise_vq: torch.Tensor,
    *,
    lambda_c: float,
    lambda_mq: float,
    lambda_vq: float,
    lambda_ce: float,
    lambda_dice: float,
    ema_alpha: float,
) -> Tuple[float, float]:
    for m in students:
        m.train()
    for t in teachers:
        t.eval()

    total_loss = 0.0
    total_dice = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        mq_soft = batch["mq_soft"].to(device)
        vq_soft = batch["vq_soft"].to(device)
        mq_weight = batch["mq_weight"].to(device)
        vq_weight = batch["vq_weight"].to(device)

        images_s, images_t = student_teacher_views(images)

        for opt in opts:
            opt.zero_grad()

        student_probs = []
        for m in students:
            logits = m(images_s)
            student_probs.append(stack_probs_from_logits(logits))
        S = torch.stack(student_probs).mean(dim=0)

        with torch.no_grad():
            teacher_probs = []
            for t in teachers:
                logits_t = t(images_t)
                teacher_probs.append(stack_probs_from_logits(logits_t))
            T = torch.stack(teacher_probs).mean(dim=0).detach()

        y_hq = to_one_hot((labels > 0.5).long(), num_classes=2)

        loss_hq_ce = weighted_ce(S, y_hq, class_weights=class_weights)
        loss_hq_dice = dice_loss(S, y_hq, class_weights=class_weights)

        ones_class = torch.ones_like(class_weights)
        loss_mq_ce = weighted_ce(
            S,
            mq_soft,
            class_weights=ones_class,
            weight_map=mq_weight,
            noise_matrix=noise_mq,
        )
        loss_mq_dice = dice_loss(
            S,
            mq_soft,
            class_weights=ones_class,
            weight_map=mq_weight,
        )

        loss_vq_ce = weighted_ce(
            S,
            vq_soft,
            class_weights=ones_class,
            weight_map=vq_weight,
            noise_matrix=noise_vq,
        )
        loss_vq_dice = dice_loss(
            S,
            vq_soft,
            class_weights=ones_class,
            weight_map=vq_weight,
        )

        loss_consistency = F.mse_loss(S, T)

        loss = (
            loss_hq_ce * lambda_ce
            + loss_hq_dice * lambda_dice
            + lambda_mq * (loss_mq_ce * lambda_ce + loss_mq_dice * lambda_dice)
            + lambda_vq * (loss_vq_ce * lambda_ce + loss_vq_dice * lambda_dice)
            + lambda_c * loss_consistency
        )
        loss.backward()

        for opt in opts:
            opt.step()

        for t, s in zip(teachers, students):
            update_ema(t, s, ema_alpha)

        total_loss += loss.item()
        prob1 = S[:, 1:2].clamp(1e-6, 1.0 - 1e-6)
        logits_for_metric = torch.log(prob1 / (1.0 - prob1))
        total_dice += calculate_dice_score(logits_for_metric, labels)

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate_ensemble(students: List[nn.Module], loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    for m in students:
        m.eval()
    total_loss = 0.0
    total_dice = 0.0
    class_weights = torch.ones(2, device=device)
    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = [m(images) for m in students]
        probs = torch.stack([stack_probs_from_logits(l) for l in logits]).mean(dim=0)
        y_hq = to_one_hot((labels > 0.5).long(), num_classes=2)
        loss = weighted_ce(probs, y_hq, class_weights=class_weights) + dice_loss(
            probs, y_hq, class_weights=class_weights
        )
        total_loss += loss.item()
        prob1 = probs[:, 1:2].clamp(1e-6, 1.0 - 1e-6)
        logits_for_metric = torch.log(prob1 / (1.0 - prob1))
        total_dice += calculate_dice_score(logits_for_metric, labels)
    n = len(loader)
    return total_loss / n, total_dice / n


if __name__ == "__main__":
    args = parse_args()
    train_mtcl(args)
