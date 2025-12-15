"""
Leader-follower fine-tuning with CL + SSDM.

Process:
1) Load 10 pre-trained UNet3D models (students), adapt 1->2 channel heads if needed.
2) Select a leader via Dice on a subset of train HQ cases (teachers, full-volume sliding window).
3) For each follower (j != leader):
   - Generate noisy hard pseudo labels on the low-quality set using follower teacher (no Frangi).
   - Run leader EMA teacher on Frangi-weighted CT to get probabilities.
   - Apply Confident Learning thresholds + SSDM relabeling to produce soft corrected labels (in RAM).
   - Fine-tune only the leader for K epochs on mixed HQ (train) + SSDM pseudo (low) with CE+Dice+consistency.
4) Output: fine-tuned leader student and EMA teacher.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.unet_model import UNet3D


# ------------------------------- I/O helpers ------------------------------- #
def _strip_extensions(name: str) -> str:
    prev = name
    while True:
        stem = Path(prev).stem
        if stem == prev or stem == "":
            return prev
        prev = stem


def _normalize_image_id(name: str) -> str:
    base = _strip_extensions(name)
    if base.startswith("image_"):
        return base
    if base.startswith("label_"):
        suffix = base.split("_", 1)[1]
        return f"image_{suffix}"
    if base.isdigit():
        return f"image_{int(base):03d}"
    return base


def read_id_list(path: Path) -> List[str]:
    with open(path, "r") as f:
        return [_normalize_image_id(line.strip()) for line in f if line.strip()]


def load_nifti_array(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img.affine


def find_matching_file(root: Path, base_id: str) -> Path | None:
    suffix = base_id.split("_", 1)[1] if "_" in base_id else base_id
    candidates = [
        root / f"{base_id}.nii.gz",
        root / f"{base_id}.nii",
        root / f"label_{suffix}.nii.gz",
        root / f"label_{suffix}.nii",
        root / f"image_{suffix}.nii.gz",
        root / f"image_{suffix}.nii",
        root / f"frangi_{base_id}.nii.gz",
        root / f"frangi_{base_id}_label.nii.gz",
        root / f"frangi_{suffix}_label.nii.gz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def normalize_ct(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-6)
    return arr.astype(np.float32)


def random_crop(arrays: List[np.ndarray], patch_size: Tuple[int, int, int]) -> List[np.ndarray]:
    pd, ph, pw = patch_size
    d, h, w = arrays[0].shape
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        arrays = [
            np.pad(a, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
            for a in arrays
        ]
        d, h, w = arrays[0].shape
    d0 = random.randint(0, d - pd) if d > pd else 0
    h0 = random.randint(0, h - ph) if h > ph else 0
    w0 = random.randint(0, w - pw) if w > pw else 0
    slices = (slice(d0, d0 + pd), slice(h0, h0 + ph), slice(w0, w0 + pw))
    return [a[slices] for a in arrays]


def _ensure_shape(arr: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Pad or crop a 3D (D,H,W) or 4D (C,D,H,W) array to exactly patch_size spatially.
    Pads with zeros if smaller; center-crops if larger.
    """
    if arr.ndim == 3:
        arr_c = arr
        prepend_shape = ()
    elif arr.ndim == 4:
        arr_c = arr
        prepend_shape = (arr.shape[0],)
    else:
        return arr

    def pad_or_crop(data: np.ndarray, target: int, axis: int) -> np.ndarray:
        cur = data.shape[axis]
        if cur == target:
            return data
        if cur < target:
            pad_before = 0
            pad_after = target - cur
            pad_width = [(0, 0)] * data.ndim
            pad_width[axis] = (pad_before, pad_after)
            data = np.pad(data, pad_width, mode="constant")
        else:
            start = 0
            data = np.take(data, indices=range(start, start + target), axis=axis)
        return data

    arr_c = pad_or_crop(arr_c, patch_size[0], -3)
    arr_c = pad_or_crop(arr_c, patch_size[1], -2)
    arr_c = pad_or_crop(arr_c, patch_size[2], -1)
    return arr_c


# ------------------------------- Dataset ----------------------------------- #
class VesselPatchDataset(Dataset):
    """
    Loads CT, HQ labels, Frangi priors, and optional pseudo labels (soft).
    Returns aligned random patches with consistent geometry.
    """

    def __init__(
        self,
        data_root: Path,
        ids: Iterable[str],
        patch_size: Tuple[int, int, int] | None,
        *,
        source: str,
        pseudo_map: Dict[str, np.ndarray] | None = None,
        use_soft_pseudo: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.ids = [_normalize_image_id(i) for i in ids]
        self.patch_size = patch_size
        self.source = source
        self.pseudo_map = pseudo_map
        self.use_soft_pseudo = use_soft_pseudo

        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.frangi_dir = self.data_root / "frangi"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        cid = self.ids[idx]
        img_path = find_matching_file(self.images_dir, cid)
        lbl_path = find_matching_file(self.labels_dir, cid)
        frangi_path = find_matching_file(self.frangi_dir, cid)
        if img_path is None or lbl_path is None or frangi_path is None:
            raise FileNotFoundError(f"Missing image/label/frangi for id {cid}")

        image, _ = load_nifti_array(img_path)
        label, _ = load_nifti_array(lbl_path)
        frangi, _ = load_nifti_array(frangi_path)

        pseudo = None
        if self.pseudo_map is not None and cid in self.pseudo_map:
            pseudo = self.pseudo_map[cid]
            if pseudo is not None and (pseudo.size == 0 or (pseudo.ndim > 0 and pseudo.shape[0] == 0)):
                # Drop empty pseudo to avoid collate errors
                pseudo = None

        if self.patch_size is not None:
            arrays = [image, label, frangi]
            if pseudo is not None:
                arrays.append(pseudo)
            cropped = random_crop(arrays, self.patch_size)
            image, label, frangi = cropped[:3]
            if pseudo is not None:
                pseudo = cropped[3]
        # Enforce exact patch size to avoid collate mismatches
        if self.patch_size is not None:
            image = _ensure_shape(image, self.patch_size)
            frangi = _ensure_shape(frangi, self.patch_size)
            label = _ensure_shape(label, self.patch_size)
            # pseudo is handled after onehot/validation below

        image = normalize_ct(np.ascontiguousarray(image))
        frangi = np.clip(frangi, 0.0, 1.0).astype(np.float32, copy=False)
        frangi = np.ascontiguousarray(frangi)
        label = (label > 0.5).astype(np.float32, copy=False)
        label = np.ascontiguousarray(label)
        label_onehot = np.stack([1.0 - label, label], axis=0).astype(np.float32, copy=False)
        label_onehot = np.ascontiguousarray(label_onehot)

        image_t = torch.from_numpy(image[None]).contiguous()
        label_t = torch.from_numpy(label_onehot).contiguous()
        frangi_t = torch.from_numpy(frangi[None]).contiguous()

        sample = {
            "id": cid,
            "image": image_t,
            "target": label_t,
            "frangi": frangi_t,
            "source": self.source,
            "has_pseudo": False,
        }
        if pseudo is not None:
            if pseudo.ndim == 3:
                pseudo = (pseudo > 0.5).astype(np.float32, copy=False)
                pseudo = np.stack([1.0 - pseudo, pseudo], axis=0)
            # Guard against degenerate or empty pseudo
            if pseudo.shape[0] < 2:
                pseudo = None
            else:
                pseudo = _ensure_shape(pseudo, self.patch_size) if self.patch_size is not None else pseudo
                pseudo = np.ascontiguousarray(pseudo.astype(np.float32))
                sample["target"] = torch.from_numpy(pseudo).contiguous()
                sample["has_pseudo"] = True
        # Final sanity check: targets must have 2 channels
        if sample["target"].shape[0] != 2:
            raise ValueError(f"Target channel mismatch for id {cid}: got shape {sample['target'].shape}")
        return sample


# ---------------------------- Loss components ------------------------------ #
def to_one_hot(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long().squeeze(1)
    oh = F.one_hot(labels, num_classes=2).permute(0, 4, 1, 2, 3).float()
    return oh


def weighted_cross_entropy(probs: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
    log_p = torch.log(probs)
    weights = class_weights.view(1, -1, 1, 1, 1)
    loss = -(weights * targets * log_p).sum(dim=1)
    return loss.mean()


def weighted_dice_loss(probs: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = class_weights.view(1, -1, 1, 1, 1)
    intersection = (weights * probs * targets).sum()
    denom = (weights * probs).sum() + (weights * targets).sum()
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice


# --------------------------- Augmentations --------------------------------- #
def photometric_augment(x: torch.Tensor, brightness: float = 0.1, contrast: float = 0.1, noise_std: float = 0.03) -> torch.Tensor:
    out = x
    if brightness > 0:
        delta = (torch.rand(x.size(0), 1, 1, 1, 1, device=x.device) - 0.5) * 2 * brightness
        out = out + delta
    if contrast > 0:
        scale = 1.0 + (torch.rand(x.size(0), 1, 1, 1, 1, device=x.device) - 0.5) * 2 * contrast
        out = out * scale
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return torch.clamp(out, 0.0, 1.0)


def frangi_weighted_view(ct: torch.Tensor, frangi: torch.Tensor, beta: float) -> torch.Tensor:
    mean_v = frangi.mean(dim=(2, 3, 4), keepdim=True)
    centered = frangi - mean_v
    weighted = ct * (1.0 + beta * centered)
    return torch.clamp(weighted, 0.0, 1.0)


# --------------------------- Model helpers --------------------------------- #
def load_student_checkpoint(model: nn.Module, path: Path, device: torch.device) -> Tuple[List[str], List[str]]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model_state = model.state_dict()
    filtered_state = {}
    shape_skips: List[str] = []
    head_w = "outc.conv.weight"
    head_b = "outc.conv.bias"
    if head_w in state and head_b in state and head_w in model_state and head_b in model_state:
        if state[head_w].shape[0] == 1 and model_state[head_w].shape[0] == 2:
            new_w = model_state[head_w].clone()
            new_b = model_state[head_b].clone()
            new_w.zero_()
            new_b.zero_()
            new_w[1] = state[head_w][0]
            new_b[1] = state[head_b][0]
            filtered_state[head_w] = new_w
            filtered_state[head_b] = new_b
            print(f"[Init] Expanded 1->2 channel head for {path.name}")
    for k, v in state.items():
        if k in filtered_state:
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            shape_skips.append(k)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if shape_skips:
        print(f"[Init] Skipped {len(shape_skips)} mismatched params (e.g., {shape_skips[:2]}) from {path.name}")
    return missing, unexpected


def load_all_models(num_models: int, device: torch.device, init_root: Path, init_name: str) -> List[nn.Module]:
    models: List[nn.Module] = []
    for idx in range(num_models):
        m = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
        ckpt_dir = init_root / f"model_{idx}"
        candidates = [ckpt_dir / init_name, ckpt_dir / "latest.pth", ckpt_dir / "best.pth"]
        loaded = False
        for cand in candidates:
            if cand.exists():
                load_student_checkpoint(m, cand, device)
                loaded = True
                break
        if not loaded:
            print(f"[Init] Model {idx}: no checkpoint found in {ckpt_dir}")
        models.append(m)
    return models


# ---------------------------- Inference utils ------------------------------ #
@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    overlap: float = 0.5,
    patch_batch_size: int = 1,
) -> torch.Tensor:
    model.eval()
    _, _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    out_sum = torch.zeros((1, 2, D, H, W), device=device)
    weight = torch.zeros((1, 1, D, H, W), device=device)

    coords = []
    max_d = max(D - pd, 0)
    max_h = max(H - ph, 0)
    max_w = max(W - pw, 0)
    for d0 in range(0, max_d + 1, sd):
        sd0 = min(d0, max_d)
        d1 = sd0 + pd
        for h0 in range(0, max_h + 1, sh):
            sh0 = min(h0, max_h)
            h1 = sh0 + ph
            for w0 in range(0, max_w + 1, sw):
                sw0 = min(w0, max_w)
                w1 = sw0 + pw
                coords.append((sd0, d1, sh0, h1, sw0, w1))

    patch_batch_size = max(1, int(patch_batch_size))
    for i in range(0, len(coords), patch_batch_size):
        batch_coords = coords[i : i + patch_batch_size]
        patches = []
        for (sd0, d1, sh0, h1, sw0, w1) in batch_coords:
            patch = volume[:, :, sd0:d1, sh0:h1, sw0:w1]
            patches.append(patch)
        batch_tensor = torch.cat(patches, dim=0).to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        for b, (sd0, d1, sh0, h1, sw0, w1) in enumerate(batch_coords):
            out_sum[:, :, sd0:d1, sh0:h1, sw0:w1] += probs[b : b + 1]
            weight[:, :, sd0:d1, sh0:h1, sw0:w1] += 1.0
    return out_sum / torch.clamp(weight, min=1.0)


def dice_from_probs(probs: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> float:
    pred = (probs >= 0.5).astype(np.float32)
    label = (label > 0.5).astype(np.float32)
    inter = (pred * label).sum()
    denom = pred.sum() + label.sum()
    return float((2 * inter + eps) / (denom + eps))


# ------------------------ Class weights & metrics -------------------------- #
def compute_class_weights(label_paths: List[Path], gamma: float) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.float64)
    for lp in label_paths:
        lbl, _ = load_nifti_array(lp)
        counts[1] += (lbl > 0.5).sum()
        counts[0] += (lbl <= 0.5).sum()
    freqs = counts / counts.sum().clamp(min=1.0)
    weights = (freqs + 1e-6).pow(-gamma)
    weights = weights / weights.sum()
    return weights.float()


# ---------------------------- Test prediction ----------------------------- #
@torch.no_grad()
def predict_on_ids(
    model: nn.Module,
    ids: List[str],
    data_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    out_dir: Path,
):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in tqdm(ids, desc="Predict test", leave=False):
        img_path = find_matching_file(data_dir / "images", cid)
        if img_path is None:
            continue
        image, affine = load_nifti_array(img_path)
        image = normalize_ct(image)
        vol = torch.from_numpy(np.ascontiguousarray(image)[None, None]).to(device)
        probs = sliding_window_predict(model, vol, patch_size, device, patch_batch_size=8)[0].cpu().numpy()
        hard = np.argmax(probs, axis=0).astype(np.uint8)
        nib.save(nib.Nifti1Image(hard, affine), out_dir / f"{cid}.nii.gz")


# ---------------------------- CL + SSDM utils ------------------------------ #
def compute_thresholds(noisy_label: np.ndarray, teacher_probs: np.ndarray, percentile: float) -> Tuple[float, float]:
    # noisy_label: [D,H,W] int {0,1}; teacher_probs: [2,D,H,W]
    taus = []
    for c in (0, 1):
        mask = noisy_label == c
        if mask.any():
            vals = teacher_probs[c][mask]
            q = np.percentile(vals, percentile * 100.0)
            taus.append(float(q))
        else:
            taus.append(0.5)
    return taus[0], taus[1]


def confident_joint_and_masks(noisy_label: np.ndarray, teacher_probs: np.ndarray, tau0: float, tau1: float):
    argmax_t = np.argmax(teacher_probs, axis=0)
    conf0 = (argmax_t == 0) & (teacher_probs[0] >= tau0)
    conf1 = (argmax_t == 1) & (teacher_probs[1] >= tau1)
    cj = np.zeros((2, 2), dtype=np.float64)
    for k in (0, 1):
        mask_k = noisy_label == k
        cj[0, k] = (conf0 & mask_k).sum()
        cj[1, k] = (conf1 & mask_k).sum()
    return cj, conf0, conf1


def ssdm_relabel(
    noisy_onehot: np.ndarray,
    teacher_probs: np.ndarray,
    tau0: float,
    tau1: float,
    kappa_bg: float,
    lambda_tau: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # noisy_onehot: [2,D,H,W]; teacher_probs: [2,D,H,W]
    noisy_label = np.argmax(noisy_onehot, axis=0)
    cj, conf0, conf1 = confident_joint_and_masks(noisy_label, teacher_probs, tau0, tau1)
    ssdm = noisy_onehot.copy()
    U1 = (noisy_label == 1).sum()

    def relabel_pair(i: int, k: int, conf_mask: np.ndarray):
        nonlocal ssdm
        mask_k = noisy_label == k
        candidates = conf_mask & mask_k
        if not candidates.any():
            return
        quota = min(int(cj[i, k]), int(candidates.sum()))
        if k == 0 and i == 1 and U1 > 0:
            quota = min(quota, int(kappa_bg * U1))
        if quota <= 0:
            return
        scores = teacher_probs[i][candidates]
        if scores.size <= quota:
            relabel_mask = candidates
        else:
            thresh = np.partition(scores, -quota)[-quota]
            relabel_mask = candidates & (teacher_probs[i] >= thresh)
        mix = np.zeros_like(ssdm)
        mix[i] = 1.0
        mix[k] = 0.0
        ssdm[:, relabel_mask] = (1 - lambda_tau) * noisy_onehot[:, relabel_mask] + lambda_tau * mix[:, relabel_mask]

    relabel_pair(0, 1, conf0)
    relabel_pair(1, 0, conf1)

    cj_eps = cj + 1.0
    noise_mat = np.zeros((2, 2), dtype=np.float32)
    for i in (0, 1):
        noise_mat[i] = cj_eps[i] / cj_eps[i].sum()
    return ssdm, noise_mat


# ------------------------ Leader selection & pseudo ------------------------ #
def select_leader(models: List[nn.Module], train_ids: List[str], data_dir: Path, patch_size: Tuple[int, int, int], device: torch.device, max_cases: int, batch_size: int) -> Tuple[int, List[float]]:
    scores: List[float] = []
    subset = train_ids if max_cases is None or len(train_ids) <= max_cases else random.sample(train_ids, max_cases)
    for idx, m in enumerate(models):
        dices = []
        for cid in tqdm(subset, desc=f"Leader eval model {idx}", leave=False):
            img_path = find_matching_file(data_dir / "images", cid)
            lbl_path = find_matching_file(data_dir / "labels", cid)
            if img_path is None or lbl_path is None:
                continue
            image, _ = load_nifti_array(img_path)
            label, _ = load_nifti_array(lbl_path)
            image = normalize_ct(image)
            vol = torch.from_numpy(image[None, None]).to(device)
            probs = sliding_window_predict(m, vol, patch_size, device, patch_batch_size=batch_size)[0].cpu().numpy()
            dice = dice_from_probs(probs[1], label)
            dices.append(dice)
        avg = float(np.mean(dices)) if dices else 0.0
        scores.append(avg)
    leader_idx = int(np.argmax(scores))
    return leader_idx, scores


def generate_noisy_map(follower: nn.Module, val_ids: List[str], data_dir: Path, patch_size: Tuple[int, int, int], device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
    noisy: Dict[str, np.ndarray] = {}
    for cid in tqdm(val_ids, desc="Follower noisy pseudo", leave=False):
        img_path = find_matching_file(data_dir / "images", cid)
        if img_path is None:
            continue
        image, _ = load_nifti_array(img_path)
        image = normalize_ct(np.ascontiguousarray(image))
        vol = torch.from_numpy(image[None, None]).to(device)
        probs = sliding_window_predict(follower, vol, patch_size, device, patch_batch_size=batch_size)[0].cpu().numpy()
        hard = np.argmax(probs, axis=0).astype(np.uint8)
        onehot = np.stack([1.0 - hard, hard], axis=0).astype(np.float32, copy=False)
        noisy[cid] = np.ascontiguousarray(onehot)
    return noisy


def generate_leader_probs_frangi(teacher: nn.Module, val_ids: List[str], data_dir: Path, patch_size: Tuple[int, int, int], device: torch.device, beta: float, batch_size: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cid in tqdm(val_ids, desc="Leader Frangi probs", leave=False):
        img_path = find_matching_file(data_dir / "images", cid)
        frangi_path = find_matching_file(data_dir / "frangi", cid)
        if img_path is None or frangi_path is None:
            continue
        image, _ = load_nifti_array(img_path)
        frangi, _ = load_nifti_array(frangi_path)
        image = normalize_ct(np.ascontiguousarray(image))
        frangi = np.ascontiguousarray(frangi)
        mean_v = frangi.mean()
        frg = np.clip(image * (1.0 + beta * (frangi - mean_v)), 0.0, 1.0).astype(np.float32, copy=False)
        frg = np.ascontiguousarray(frg)
        vol = torch.from_numpy(frg[None, None]).to(device)
        probs = sliding_window_predict(teacher, vol, patch_size, device, patch_batch_size=batch_size)[0].cpu().numpy()
        out[cid] = np.ascontiguousarray(probs)
    return out


def build_ssdm_map(
    noisy_map: Dict[str, np.ndarray],
    leader_probs: Dict[str, np.ndarray],
    tau_percentile: float,
    lambda_tau: float,
    kappa_bg: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ssdm_map: Dict[str, np.ndarray] = {}
    noise_mats: Dict[str, np.ndarray] = {}
    for cid in tqdm(noisy_map.keys(), desc="SSDM relabel", leave=False):
        noisy = noisy_map[cid]
        teacher_p = leader_probs.get(cid)
        if teacher_p is None:
            continue
        noisy_label = np.argmax(noisy, axis=0)
        tau0, tau1 = compute_thresholds(noisy_label, teacher_p, tau_percentile)
        ssdm, noise_mat = ssdm_relabel(noisy, teacher_p, tau0, tau1, kappa_bg, lambda_tau)
        ssdm_map[cid] = ssdm.astype(np.float32)
        noise_mats[cid] = noise_mat
    return ssdm_map, noise_mats


# --------------------------- Training primitives --------------------------- #
def leader_batch_loss(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict,
    class_weights: torch.Tensor,
    cfg: "MTCLConfig",
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = batch["image"].to(device)
    frangi = batch["frangi"].to(device)
    targets = batch["target"].to(device)
    sources = batch["source"]

    x_s = photometric_augment(imgs)
    x_t = frangi_weighted_view(imgs, frangi, beta=cfg.beta_frangi)
    x_t = photometric_augment(x_t)

    student_logits = student(x_s)
    s_probs = torch.softmax(student_logits, dim=1)

    with torch.no_grad():
        t_logits = teacher(x_t)
        t_probs = torch.softmax(t_logits, dim=1)

    train_mask = torch.tensor([src == "train" for src in sources], device=device, dtype=torch.bool)
    low_mask = torch.tensor([src == "low" for src in sources], device=device, dtype=torch.bool)

    loss_hq = torch.tensor(0.0, device=device)
    loss_low = torch.tensor(0.0, device=device)

    if train_mask.any():
        tgt_hq = targets[train_mask]
        ce = weighted_cross_entropy(s_probs[train_mask], tgt_hq, class_weights)
        dice = weighted_dice_loss(s_probs[train_mask], tgt_hq, class_weights)
        loss_hq = cfg.lambda_ce * ce + cfg.lambda_dice * dice

    if low_mask.any():
        tgt_low = targets[low_mask]
        ce = weighted_cross_entropy(s_probs[low_mask], tgt_low, class_weights)
        dice = weighted_dice_loss(s_probs[low_mask], tgt_low, class_weights)
        loss_low = cfg.lambda_ce * ce + cfg.lambda_dice * dice

    loss_cons = F.mse_loss(s_probs, t_probs)
    total = loss_hq + cfg.lambda_pl * loss_low + cfg.lambda_c * loss_cons
    return total, loss_hq.detach(), loss_low.detach()


def train_leader_epoch(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: "MTCLConfig",
    device: torch.device,
) -> dict:
    student.train()
    teacher.eval()
    tot = hq = low = 0.0
    n = len(loader)
    for batch in tqdm(loader, desc="Leader train", leave=False):
        loss, lhq, llow = leader_batch_loss(student, teacher, batch, class_weights, cfg, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for t_p, s_p in zip(teacher.parameters(), student.parameters()):
                t_p.data.mul_(cfg.ema_alpha).add_(s_p.data, alpha=1.0 - cfg.ema_alpha)
        tot += loss.item()
        hq += lhq.item()
        low += llow.item()
    return {
        "loss": tot / max(n, 1),
        "hq_loss": hq / max(n, 1),
        "low_loss": low / max(n, 1),
    }


# ----------------------------- Config & args ------------------------------- #
@dataclass
class MTCLConfig:
    data_dir: Path
    output_dir: Path
    train_split: Path
    val_split: Path | None
    test_split: Path | None
    val_fraction: float
    batch_size: int
    num_workers: int
    lr: float
    ema_alpha: float
    lambda_ce: float
    lambda_dice: float
    lambda_pl: float
    lambda_c: float
    beta_frangi: float
    gamma_class: float
    patch_size: Tuple[int, int, int]
    eval_patch_size: Tuple[int, int, int]
    eval_batch_size: int
    num_models: int
    device: str
    pseudo_soft: bool
    save_checkpoints: bool
    init_checkpoint_root: Path
    init_checkpoint_name: str
    leader_idx: int
    leader_epochs_per_follower: int
    tau_percentile: float
    lambda_tau: float
    kappa_bg: float
    max_train_eval: int


def parse_args() -> MTCLConfig:
    p = argparse.ArgumentParser(description="Leader-follower fine-tuning with CL+SSDM.")
    p.add_argument("--data_dir", type=Path, default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data"))
    p.add_argument("--output_dir", type=Path, default=Path("./results/output_mtcl_finetune"))
    p.add_argument("--train_split", type=Path, default=Path("./data/splits/train_list.txt"))
    p.add_argument("--val_split", type=Path, default=None)
    p.add_argument("--test_split", type=Path, default=Path("./data/splits/test_list.txt"))
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_alpha", type=float, default=0.99)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_dice", type=float, default=1.0)
    p.add_argument("--lambda_pl", type=float, default=0.5)
    p.add_argument("--lambda_c", type=float, default=0.3)
    p.add_argument("--beta_frangi", type=float, default=0.5)
    p.add_argument("--gamma_class", type=float, default=0.5)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--eval_patch_size", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_models", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--pseudo_soft", action="store_true", help="Keep soft pseudo labels (else hard).")
    p.add_argument("--save_checkpoints", action="store_true", default=True)
    p.add_argument("--init_checkpoint_root", type=Path, default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_ensemble/checkpoints"))
    p.add_argument("--init_checkpoint_name", type=str, default="best.pth")
    p.add_argument("--leader_idx", type=int, default=9, help="Fixed leader model index (0-based).")
    p.add_argument("--leader_epochs_per_follower", type=int, default=10)
    p.add_argument("--tau_percentile", type=float, default=0.85)
    p.add_argument("--lambda_tau", type=float, default=0.8)
    p.add_argument("--kappa_bg", type=float, default=0.1)
    p.add_argument("--max_train_eval", type=int, default=10)
    args = p.parse_args()
    return MTCLConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        ema_alpha=args.ema_alpha,
        lambda_ce=args.lambda_ce,
        lambda_dice=args.lambda_dice,
        lambda_pl=args.lambda_pl,
        lambda_c=args.lambda_c,
        beta_frangi=args.beta_frangi,
        gamma_class=args.gamma_class,
        patch_size=tuple(args.patch_size),
        eval_patch_size=tuple(args.eval_patch_size),
        eval_batch_size=args.eval_batch_size,
        num_models=args.num_models,
        device=args.device,
        pseudo_soft=args.pseudo_soft,
        save_checkpoints=args.save_checkpoints,
        init_checkpoint_root=args.init_checkpoint_root,
        init_checkpoint_name=args.init_checkpoint_name,
        leader_idx=args.leader_idx,
        leader_epochs_per_follower=args.leader_epochs_per_follower,
        tau_percentile=args.tau_percentile,
        lambda_tau=args.lambda_tau,
        kappa_bg=args.kappa_bg,
        max_train_eval=args.max_train_eval,
    )


# ------------------------------- Main driver ------------------------------- #
def prepare_splits(cfg: MTCLConfig) -> Tuple[List[str], List[str]]:
    train_ids_full = read_id_list(cfg.train_split)
    if cfg.val_split is not None and cfg.val_split.exists():
        val_ids = read_id_list(cfg.val_split)
        train_ids = [cid for cid in train_ids_full if cid not in set(val_ids)]
    else:
        rng = random.Random(0)
        ids = train_ids_full.copy()
        rng.shuffle(ids)
        val_count = max(1, int(len(ids) * cfg.val_fraction))
        val_ids = ids[:val_count]
        train_ids = ids[val_count:]
        split_dir = cfg.train_split.parent
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "val_from_train.txt", "w") as f:
            for cid in val_ids:
                f.write(f"{cid}\n")
        with open(split_dir / "train_from_train.txt", "w") as f:
            for cid in train_ids:
                f.write(f"{cid}\n")
    return train_ids, val_ids


def freeze_all(models: List[nn.Module]):
    for m in models:
        for p in m.parameters():
            p.requires_grad_(False)


def train_mtcl(cfg: MTCLConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = cfg.output_dir / "history.json"
    ckpt_dir = cfg.output_dir / "checkpoints"

    train_ids, val_ids = prepare_splits(cfg)
    label_paths = [find_matching_file(cfg.data_dir / "labels", cid) for cid in train_ids]
    label_paths = [p for p in label_paths if p is not None]
    class_weights = compute_class_weights(label_paths, cfg.gamma_class).to(device)

    # Load all models, fixed leader index
    models = load_all_models(cfg.num_models, device, cfg.init_checkpoint_root, cfg.init_checkpoint_name)
    leader_idx = cfg.leader_idx
    print(f"[Leader] Using fixed leader model {leader_idx}")

    # Build leader student/teacher
    leader_student = models[leader_idx]
    leader_teacher = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
    leader_teacher.load_state_dict(leader_student.state_dict())

    # Freeze followers
    followers = [models[i] for i in range(cfg.num_models) if i != leader_idx]
    freeze_all(followers)

    optimizer = torch.optim.Adam(leader_student.parameters(), lr=cfg.lr)
    history = []

    # Iterate followers
    for follower_idx, follower in enumerate(followers):
        print(f"[Follower {follower_idx}] Processing follower model")
        follower.eval()
        # Noisy pseudo from follower teacher (use its own weights as teacher)
        noisy_map = generate_noisy_map(follower, val_ids, cfg.data_dir, cfg.eval_patch_size, device, cfg.eval_batch_size)
        # Leader probs on Frangi-weighted CT
        leader_probs = generate_leader_probs_frangi(leader_teacher, val_ids, cfg.data_dir, cfg.eval_patch_size, device, cfg.beta_frangi, cfg.eval_batch_size)
        # CL + SSDM
        ssdm_map, noise_mats = build_ssdm_map(noisy_map, leader_probs, cfg.tau_percentile, cfg.lambda_tau, cfg.kappa_bg)

        for e in range(1, cfg.leader_epochs_per_follower + 1):
            train_ds = VesselPatchDataset(cfg.data_dir, train_ids, cfg.patch_size, source="train", pseudo_map=None)
            low_ds = VesselPatchDataset(cfg.data_dir, val_ids, cfg.patch_size, source="low", pseudo_map=ssdm_map, use_soft_pseudo=True)
            mixed = ConcatDataset([train_ds, low_ds])
            loader = DataLoader(mixed, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
            metrics = train_leader_epoch(leader_student, leader_teacher, optimizer, loader, class_weights, cfg, device)
            print(f"[Follower {follower_idx}] Epoch {e}/{cfg.leader_epochs_per_follower} loss={metrics['loss']:.4f} hq={metrics['hq_loss']:.4f} low={metrics['low_loss']:.4f}")
            history.append({
                "follower_idx": follower_idx,
                "epoch": e,
                "metrics": metrics,
            })
            if cfg.save_checkpoints:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "leader_idx": leader_idx,
                    "follower_idx": follower_idx,
                    "epoch": e,
                    "student": leader_student.state_dict(),
                    "teacher": leader_teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                }, ckpt_dir / f"follower_{follower_idx:02d}_epoch_{e:02d}.pth")

        if cfg.save_checkpoints:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "leader_idx": leader_idx,
                "follower_idx": follower_idx,
                "student": leader_student.state_dict(),
                "teacher": leader_teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            }, ckpt_dir / f"follower_{follower_idx:02d}.pth")

        # Save intermediate history
        with open(history_path, "w") as f:
            import json
            json.dump(history, f, indent=2)

    # Final checkpoint
    if cfg.save_checkpoints:
        torch.save({
            "leader_idx": leader_idx,
            "student": leader_student.state_dict(),
            "teacher": leader_teacher.state_dict(),
            "history": history,
        }, ckpt_dir / "leader_final.pth")
    print("[Done] Fine-tuning complete.")

    # Optional: predict on test_split if provided
    if cfg.test_split is not None and cfg.test_split.exists():
        test_ids = read_id_list(cfg.test_split)
        preds_dir = cfg.output_dir / "predictions_test"
        predict_on_ids(
            leader_teacher,
            test_ids,
            cfg.data_dir,
            cfg.eval_patch_size,
            device,
            cfg.beta_frangi,
            preds_dir,
        )
        print(f"[Predict] Saved test predictions to {preds_dir}")


def main():
    cfg = parse_args()
    train_mtcl(cfg)


if __name__ == "__main__":
    main()
