"""
Single student + EMA teacher fine-tuning with:
- Noisy labels from an ensemble mean prediction volume (per-case, per-voxel probability or mask).
- Confident Learning (binary) to locate likely-noisy voxels.
- SSDM (scheme 2): shrink noisy-label probabilities toward the EMA teacher prediction.
- Frangi is injected by ADDING a small-weight Frangi map directly to the CT input (still 1-channel).

Expected filenames (example case "image_001"):
  images:   data/images/image_001.nii.gz
  labels:   data/labels/label_001.nii.gz
  frangi:   data/frangi_lungmasked/pred_frangi_lungmasked_image_001.nii.gz
  ensemble: data/unet_prediction/ensemble_lungmasked/pred_001.nii.gz

  python /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/model_script/train/mtcl_single_best_frangiadd.py \
  --data_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data \
  --ensemble_pred_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/ensemble_lungmasked \
  --init_checkpoint /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_ensemble/checkpoints/model_9/best.pth \
  --frangi_eps 0.01 \
  --tau_ssdm 0.8 \
  --lambda_cl_max 0.5 \
  --lambda_cl_warmup_iters 4000 \
  --batch_size 2 \
  --num_epochs 50


Notes:
- CT normalization: clip HU to [-1000, 400] then map to [0,1].
- Frangi normalization: clip to [0,1] (or robust min-max) then add to CT with weight --frangi_eps.
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
from torch.utils.data import DataLoader, Dataset
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
    """
    Convert various id formats to "image_###".
    Accepts:
      - "image_001", "image_001.nii.gz"
      - "label_001", "001", "pred_001"
    """
    base = _strip_extensions(name)
    if base.startswith("image_"):
        return base
    if base.startswith("label_"):
        suffix = base.split("_", 1)[1]
        return f"image_{suffix}"
    if base.startswith("pred_"):
        suffix = base.split("_", 1)[1]
        if suffix.isdigit():
            return f"image_{int(suffix):03d}"
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
    """
    Find corresponding file under root for a normalized id like "image_001".
    Includes patterns for:
      - images: image_001.nii.gz
      - labels: label_001.nii.gz
      - frangi: pred_frangi_lungmasked_image_001.nii.gz
      - ensemble preds: pred_001.nii.gz
    """
    suffix = base_id.split("_", 1)[1] if "_" in base_id else base_id
    candidates = [
        # direct
        root / f"{base_id}.nii.gz",
        root / f"{base_id}.nii",
        # label
        root / f"label_{suffix}.nii.gz",
        root / f"label_{suffix}.nii",
        # image
        root / f"image_{suffix}.nii.gz",
        root / f"image_{suffix}.nii",
        # ensemble
        root / f"pred_{suffix}.nii.gz",
        root / f"pred_{suffix}.nii",
        # frangi 
        root / f"pred_frangi_lungmasked_{base_id}.nii.gz",
        root / f"pred_frangi_lungmasked_{base_id}.nii",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


# -------------------------- Normalization / input -------------------------- #
def normalize_ct_hu(arr: np.ndarray) -> np.ndarray:
    """Clip HU to [-1000, 400] then map to [0,1]."""
    arr = np.clip(arr, -1000.0, 400.0)
    arr = (arr + 1000.0) / 1400.0
    return arr.astype(np.float32, copy=False)


def normalize_frangi(fr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    If your frangi is already [0,1] you can keep it.
    This version does robust min-max to reduce outliers.
    """
    fr = fr.astype(np.float32, copy=False)
    lo = np.quantile(fr, 0.01)
    hi = np.quantile(fr, 0.99)
    fr = np.clip(fr, lo, hi)
    fr = (fr - fr.min()) / (fr.max() - fr.min() + eps)
    return fr.astype(np.float32, copy=False)


def random_crop(arrays: List[np.ndarray], patch_size: Tuple[int, int, int]) -> List[np.ndarray]:
    pd, ph, pw = patch_size
    d, h, w = arrays[0].shape
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        arrays = [np.pad(a, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant") for a in arrays]
        d, h, w = arrays[0].shape
    d0 = random.randint(0, d - pd) if d > pd else 0
    h0 = random.randint(0, h - ph) if h > ph else 0
    w0 = random.randint(0, w - pw) if w > pw else 0
    slices = (slice(d0, d0 + pd), slice(h0, h0 + ph), slice(w0, w0 + pw))
    return [a[slices] for a in arrays]


def _ensure_shape(arr: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Pad or crop a 3D (D,H,W) or 4D (C,D,H,W) array to exactly patch_size spatially."""
    def pad_or_crop(data: np.ndarray, target: int, axis: int) -> np.ndarray:
        cur = data.shape[axis]
        if cur == target:
            return data
        if cur < target:
            pad_width = [(0, 0)] * data.ndim
            pad_width[axis] = (0, target - cur)
            return np.pad(data, pad_width, mode="constant")
        # crop
        start = 0
        return np.take(data, indices=range(start, start + target), axis=axis)

    arr = pad_or_crop(arr, patch_size[0], -3)
    arr = pad_or_crop(arr, patch_size[1], -2)
    arr = pad_or_crop(arr, patch_size[2], -1)
    return arr


# ------------------------------- Dataset ----------------------------------- #
class VesselPatchDataset(Dataset):
    """
    Loads CT, GT label, Frangi, and a per-case pseudo probability map (ensemble mean).
    Returns random aligned patches.
    """

    def __init__(
        self,
        data_root: Path,
        ids: Iterable[str],
        patch_size: Tuple[int, int, int],
        *,
        pseudo_prob_map: Dict[str, np.ndarray] | None,
    ) -> None:
        self.data_root = Path(data_root)
        self.ids = [_normalize_image_id(i) for i in ids]
        self.patch_size = patch_size
        self.pseudo_prob_map = pseudo_prob_map

        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.frangi_dir = self.data_root / "frangi_lungmasked"

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

        pseudo_prob = None
        if self.pseudo_prob_map is not None and cid in self.pseudo_prob_map:
            pseudo_prob = self.pseudo_prob_map[cid]

        if self.patch_size is not None:
            arrays = [image, label, frangi]
            if pseudo_prob is not None:
                arrays.append(pseudo_prob)
            cropped = random_crop(arrays, self.patch_size)
            image, label, frangi = cropped[:3]
            if pseudo_prob is not None:
                pseudo_prob = cropped[3]

            image = _ensure_shape(image, self.patch_size)
            frangi = _ensure_shape(frangi, self.patch_size)
            label = _ensure_shape(label, self.patch_size)
            if pseudo_prob is not None:
                pseudo_prob = _ensure_shape(pseudo_prob, self.patch_size)

        # preprocess
        image = normalize_ct_hu(np.ascontiguousarray(image))
        frangi = normalize_frangi(np.ascontiguousarray(frangi))
        label = (label > 0.5).astype(np.float32, copy=False)

        # one-hot GT
        gt_onehot = np.stack([1.0 - label, label], axis=0).astype(np.float32, copy=False)

        # pseudo one-hot (soft)
        pseudo_onehot = None
        if pseudo_prob is not None:
            pseudo_prob = np.clip(pseudo_prob.astype(np.float32, copy=False), 0.0, 1.0)
            pseudo_onehot = np.stack([1.0 - pseudo_prob, pseudo_prob], axis=0).astype(np.float32, copy=False)

        sample = {
            "id": cid,
            "image": torch.from_numpy(image[None]).contiguous(),     # [1,D,H,W]
            "frangi": torch.from_numpy(frangi[None]).contiguous(),   # [1,D,H,W]
            "gt": torch.from_numpy(gt_onehot).contiguous(),          # [2,D,H,W]
            "has_pseudo": pseudo_onehot is not None,
        }
        if pseudo_onehot is not None:
            sample["pseudo"] = torch.from_numpy(pseudo_onehot).contiguous()  # [2,D,H,W]
        return sample


# ---------------------------- Loss components ------------------------------ #
def weighted_cross_entropy(probs: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """
    probs:   [B,2,D,H,W], softmax
    targets: [B,2,D,H,W], one-hot or soft one-hot
    """
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


# --------------------------- Augmentations --------------------------------- #
def photometric_augment(
    x: torch.Tensor,
    brightness: float = 0.1,
    contrast: float = 0.1,
    noise_std: float = 0.03
) -> torch.Tensor:
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


def add_frangi_to_ct(ct: torch.Tensor, frangi: torch.Tensor, eps: float) -> torch.Tensor:
    """Directly add a small-weight Frangi map to CT (both in [0,1])."""
    return torch.clamp(ct + eps * frangi, 0.0, 1.0)


# --------------------------- EMA / checkpoint ------------------------------ #
@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, alpha: float):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(alpha).add_(p_s.data, alpha=(1.0 - alpha))


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def load_student_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """
    Robust-ish loader:
    - tries common keys: model_state_dict / network_state_dict / state_dict / student / model
    - strips 'module.' prefix
    - if checkpoint head has 1 output channel and model expects 2, expands it to the vessel channel (index 1)
    - loads only matching shapes with strict=False
    """
    raw = torch.load(path, map_location=device)

    state = raw
    if isinstance(raw, dict):
        for key in ("model_state_dict", "network_state_dict", "state_dict", "student", "model"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {path} does not look like a state_dict.")

    state = _strip_module_prefix(state)

    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}

    # Expand 1-channel sigmoid head (ensemble pretrain) to 2-class softmax head (bg, vessel).
    head_w = "outc.conv.weight"
    head_b = "outc.conv.bias"
    if head_w in state and head_b in state and head_w in model_state and head_b in model_state:
        ckpt_w, ckpt_b = state[head_w], state[head_b]
        model_w, model_b = model_state[head_w], model_state[head_b]
        if (
            ckpt_w.ndim == model_w.ndim
            and ckpt_b.ndim == model_b.ndim
            and ckpt_w.shape[0] == 1
            and model_w.shape[0] == 2
            and ckpt_w.shape[1:] == model_w.shape[1:]
            and ckpt_b.shape == (1,)
            and model_b.shape == (2,)
        ):
            new_w = model_w.clone()
            new_b = model_b.clone()
            new_w.zero_()
            new_b.zero_()
            new_w[1] = ckpt_w[0]
            new_b[1] = ckpt_b[0]
            filtered[head_w] = new_w
            filtered[head_b] = new_b
            print(f"[Init] Expanded 1->2 channel head from {path}")

    for k, v in state.items():
        if k in filtered:
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Init] Loaded {len(filtered)} tensors from {path}")
    if missing:
        print(f"[Init] Missing keys (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[Init] Unexpected keys (showing up to 5): {unexpected[:5]}")


# -------------------------- Sliding-window inference ------------------------ #
@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    volume: torch.Tensor,                 # [1,1,D,H,W]
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
        patches = [volume[:, :, sd0:d1, sh0:h1, sw0:w1] for (sd0, d1, sh0, h1, sw0, w1) in batch_coords]
        batch_tensor = torch.cat(patches, dim=0).to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        for b, (sd0, d1, sh0, h1, sw0, w1) in enumerate(batch_coords):
            out_sum[:, :, sd0:d1, sh0:h1, sw0:w1] += probs[b : b + 1]
            weight[:, :, sd0:d1, sh0:h1, sw0:w1] += 1.0
    return out_sum / torch.clamp(weight, min=1.0)


# ---------------------- Ensemble pred loading (prob) ------------------------ #
def load_ensemble_prob_map(ids: List[str], pred_dir: Path) -> Dict[str, np.ndarray]:
    """
    Returns dict: cid -> vessel probability [D,H,W] in [0,1].
    If pred file is hard mask 0/1, it still works (just less informative for CL).
    """
    out: Dict[str, np.ndarray] = {}
    for cid in tqdm(ids, desc="Load ensemble preds", leave=False):
        pth = find_matching_file(pred_dir, cid)
        if pth is None:
            raise FileNotFoundError(f"Missing ensemble pred for {cid} under {pred_dir}")
        arr, _ = load_nifti_array(pth)
        arr = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
        out[cid] = np.ascontiguousarray(arr)
    return out


# ----------------------------- CL (binary) --------------------------------- #
def cl_noise_mask_binary(
    p_hat: np.ndarray,           # [D,H,W] prob of class-1 (ensemble mean)
    y_tilde: np.ndarray,         # [D,H,W] current soft label prob (self-loop)
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Adapted Confident Learning (binary) to produce a noisy-voxel mask.

    Steps:
      - observed label: y_obs = 1[y_tilde >= 0.5]
      - thresholds:
          t1 = mean(p_hat | y_obs==1) ; t0 = mean(1-p_hat | y_obs==0)
      - predicted latent label y* only for voxels passing class thresholds
      - confusion C[i,j] counts (observed i, latent j)
      - joint Q via row-normalization + reweight + global normalize (Eq.2)
      - noise rate in each observed class: eta_i = sum_{j!=i} Q[i,j] / sum_j Q[i,j]
      - PBC: pick k_i = round(eta_i * n_i) lowest self-confidence in class i
    """
    p_hat = np.clip(p_hat.astype(np.float32, copy=False), 0.0, 1.0)
    y_tilde = np.clip(y_tilde.astype(np.float32, copy=False), 0.0, 1.0)

    y_obs = (y_tilde >= 0.5).astype(np.int32)  # 0/1

    p1 = p_hat
    p0 = 1.0 - p_hat

    mask1 = y_obs == 1
    mask0 = y_obs == 0

    t1 = float(p1[mask1].mean()) if mask1.any() else 1.0
    t0 = float(p0[mask0].mean()) if mask0.any() else 1.0

    cand1 = p1 >= t1
    cand0 = p0 >= t0

    y_star = np.full_like(y_obs, -1, dtype=np.int32)
    y_star[cand1 & (~cand0)] = 1
    y_star[cand0 & (~cand1)] = 0
    both = cand0 & cand1
    y_star[both] = (p1[both] >= p0[both]).astype(np.int32)

    valid = y_star >= 0
    C = np.zeros((2, 2), dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            C[i, j] = np.logical_and.reduce([y_obs == i, y_star == j, valid]).sum()

    counts = np.array([mask0.sum(), mask1.sum()], dtype=np.float64)
    row_sums = np.maximum(C.sum(axis=1, keepdims=True), eps)
    R = C / row_sums
    W = R * counts[:, None]
    denom = max(W.sum(), eps)
    Q = W / denom

    # noise rates per observed class
    q_row = np.maximum(Q.sum(axis=1), eps)
    eta0 = Q[0, 1] / q_row[0]
    eta1 = Q[1, 0] / q_row[1]

    Xn = np.zeros_like(y_obs, dtype=bool)

    # class 0: self-confidence = p0
    n0 = int(counts[0])
    k0 = int(np.round(eta0 * n0))
    if n0 > 0 and k0 > 0:
        conf0 = p0[mask0]
        # threshold for bottom-k
        th = np.partition(conf0, k0 - 1)[k0 - 1]
        Xn[mask0] = conf0 <= th

    # class 1: self-confidence = p1
    n1 = int(counts[1])
    k1 = int(np.round(eta1 * n1))
    if n1 > 0 and k1 > 0:
        conf1 = p1[mask1]
        th = np.partition(conf1, k1 - 1)[k1 - 1]
        Xn[mask1] = conf1 <= th

    return Xn


# ------------------------- SSDM scheme 2 (shrink) --------------------------- #
def ssdm_shrink_to_teacher(
    y_tilde: np.ndarray,         # [D,H,W] current prob
    Xn: np.ndarray,              # [D,H,W] bool
    p_teacher: np.ndarray,       # [D,H,W] teacher prob
    tau: float,
) -> np.ndarray:
    y_tilde = np.clip(y_tilde.astype(np.float32, copy=False), 0.0, 1.0)
    p_teacher = np.clip(p_teacher.astype(np.float32, copy=False), 0.0, 1.0)
    Xn_f = Xn.astype(np.float32)
    y_new = y_tilde + tau * Xn_f * (p_teacher - y_tilde)
    return np.clip(y_new, 0.0, 1.0).astype(np.float32, copy=False)


@torch.no_grad()
def generate_teacher_probs(
    teacher: nn.Module,
    ids: List[str],
    data_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    frangi_eps: float,
    patch_batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Full-volume teacher probs for each case under ids.
    teacher input: clamp( CT_norm + frangi_eps * frangi_norm )
    returns dict: cid -> probs [2,D,H,W] (softmax)
    """
    out: Dict[str, np.ndarray] = {}
    for cid in tqdm(ids, desc="Teacher probs", leave=False):
        img_path = find_matching_file(data_dir / "images", cid)
        frangi_path = find_matching_file(data_dir / "frangi_lungmasked", cid)
        if img_path is None or frangi_path is None:
            raise FileNotFoundError(f"Missing image/frangi for id {cid}")
        image, _ = load_nifti_array(img_path)
        frangi, _ = load_nifti_array(frangi_path)
        image = normalize_ct_hu(np.ascontiguousarray(image))
        frangi = normalize_frangi(np.ascontiguousarray(frangi))
        x = np.clip(image + frangi_eps * frangi, 0.0, 1.0).astype(np.float32, copy=False)
        vol = torch.from_numpy(x[None, None]).to(device)
        probs = sliding_window_predict(teacher, vol, patch_size, device, patch_batch_size=patch_batch_size)[0].cpu().numpy()
        out[cid] = np.ascontiguousarray(probs)
    return out


def refine_pseudo_map_inplace(
    pseudo_map: Dict[str, np.ndarray],        # cid -> y_tilde prob [D,H,W]
    p_hat_map: Dict[str, np.ndarray],         # cid -> p_hat prob [D,H,W]  (ensemble)
    teacher_probs: Dict[str, np.ndarray],     # cid -> probs [2,D,H,W]
    tau: float,
) -> Dict[str, float]:
    """
    Update pseudo_map in-place:
      Xn = CL(p_hat, y_tilde)
      y_new = y_tilde + tau * Xn * (p_teacher - y_tilde)
    Returns simple stats: avg noisy-rate etc.
    """
    stats = {"avg_noisy_frac": 0.0}
    fracs = []
    for cid, y_tilde in pseudo_map.items():
        p_hat = p_hat_map[cid]
        p_teacher = teacher_probs[cid][1]
        Xn = cl_noise_mask_binary(p_hat=p_hat, y_tilde=y_tilde)
        frac = float(Xn.mean())
        fracs.append(frac)
        pseudo_map[cid] = ssdm_shrink_to_teacher(y_tilde, Xn, p_teacher, tau=tau)
    stats["avg_noisy_frac"] = float(np.mean(fracs)) if fracs else 0.0
    return stats


# ---------------------------- Training loop -------------------------------- #
def dice_from_probs(probs1: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> float:
    pred = (probs1 >= 0.5).astype(np.float32)
    label = (label > 0.5).astype(np.float32)
    inter = (pred * label).sum()
    denom = pred.sum() + label.sum()
    return float((2 * inter + eps) / (denom + eps))


def lambda_cl_schedule(step: int, warmup_iters: int, lambda_cl_max: float) -> float:
    return 0.0 if step < warmup_iters else float(lambda_cl_max)


def leader_batch_loss(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict,
    class_weights: torch.Tensor,
    cfg: "MTCLConfig",
    device: torch.device,
    global_step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    imgs = batch["image"].to(device)   # [B,1,D,H,W]
    frangi = batch["frangi"].to(device)
    gt = batch["gt"].to(device)

    base = add_frangi_to_ct(imgs, frangi, eps=cfg.frangi_eps)

    x_s = photometric_augment(base, noise_std=cfg.student_noise_std)
    x_t = photometric_augment(base, noise_std=cfg.teacher_noise_std)

    student_logits = student(x_s)
    s_probs = torch.softmax(student_logits, dim=1)

    with torch.no_grad():
        t_logits = teacher(x_t)
        t_probs = torch.softmax(t_logits, dim=1)

    # supervised (GT)
    ce_gt = weighted_cross_entropy(s_probs, gt, class_weights)
    dice_gt = weighted_dice_loss(s_probs, gt, class_weights)
    loss_gt = cfg.lambda_ce * ce_gt + cfg.lambda_dice * dice_gt

    # pseudo label (if provided)
    loss_pseudo = torch.tensor(0.0, device=device)
    has_pseudo = batch.get("has_pseudo", False)
    if torch.is_tensor(has_pseudo):
        has_pseudo = bool(has_pseudo.any().item())
    else:
        has_pseudo = bool(has_pseudo)
    if has_pseudo and "pseudo" in batch:
        pseudo = batch["pseudo"].to(device)
        ce_p = weighted_cross_entropy(s_probs, pseudo, class_weights)
        dice_p = weighted_dice_loss(s_probs, pseudo, class_weights)
        loss_pseudo = cfg.lambda_ce * ce_p + cfg.lambda_dice * dice_p

    # consistency
    loss_cons = F.mse_loss(s_probs, t_probs)

    lam_cl = lambda_cl_schedule(global_step, cfg.lambda_cl_warmup_iters, cfg.lambda_cl_max)
    total = loss_gt + lam_cl * loss_pseudo + cfg.lambda_c * loss_cons

    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_gt": float(loss_gt.detach().cpu()),
        "loss_pseudo": float(loss_pseudo.detach().cpu()),
        "loss_cons": float(loss_cons.detach().cpu()),
        "lambda_cl": float(lam_cl),
    }
    return total, logs


def train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: "MTCLConfig",
    device: torch.device,
    global_step: int,
) -> Tuple[dict, int]:
    student.train()
    teacher.eval()
    sums = {"loss_total": 0.0, "loss_gt": 0.0, "loss_pseudo": 0.0, "loss_cons": 0.0, "lambda_cl": 0.0}
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        loss, logs = leader_batch_loss(student, teacher, batch, class_weights, cfg, device, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_update(teacher, student, alpha=cfg.ema_alpha)

        for k in sums:
            sums[k] += logs[k]
        n += 1
        global_step += 1

    for k in sums:
        sums[k] = sums[k] / max(n, 1)
    return sums, global_step


@torch.no_grad()
def evaluate_dice(model: nn.Module, ids: List[str], data_dir: Path, patch_size: Tuple[int, int, int], device: torch.device, frangi_eps: float) -> float:
    model.eval()
    dices = []
    for cid in tqdm(ids, desc="Val", leave=False):
        img_path = find_matching_file(data_dir / "images", cid)
        lbl_path = find_matching_file(data_dir / "labels", cid)
        frangi_path = find_matching_file(data_dir / "frangi_lungmasked", cid)
        if img_path is None or lbl_path is None or frangi_path is None:
            continue
        image, _ = load_nifti_array(img_path)
        frangi, _ = load_nifti_array(frangi_path)
        label, _ = load_nifti_array(lbl_path)

        image = normalize_ct_hu(np.ascontiguousarray(image))
        frangi = normalize_frangi(np.ascontiguousarray(frangi))
        x = np.clip(image + frangi_eps * frangi, 0.0, 1.0).astype(np.float32, copy=False)
        vol = torch.from_numpy(x[None, None]).to(device)
        probs = sliding_window_predict(model, vol, patch_size, device, patch_batch_size=8)[0].cpu().numpy()
        dices.append(dice_from_probs(probs[1], label))
    return float(np.mean(dices)) if dices else 0.0


# ----------------------------- Config & args ------------------------------- #
@dataclass
class MTCLConfig:
    data_dir: Path
    ensemble_pred_dir: Path
    output_dir: Path
    train_split: Path
    val_split: Path | None
    val_fraction: float
    batch_size: int
    num_workers: int
    lr: float
    ema_alpha: float
    lambda_ce: float
    lambda_dice: float
    lambda_c: float
    lambda_cl_max: float
    lambda_cl_warmup_iters: int
    frangi_eps: float
    tau_ssdm: float
    gamma_class: float
    patch_size: Tuple[int, int, int]
    eval_patch_size: Tuple[int, int, int]
    eval_batch_size: int
    num_epochs: int
    refine_every: int
    device: str
    init_checkpoint: Path
    resume_checkpoint: Path | None
    student_noise_std: float
    teacher_noise_std: float
    save_checkpoints: bool


def parse_args() -> MTCLConfig:
    p = argparse.ArgumentParser(description="Single-model MTCL (ensemble noisy labels + CL + SSDM shrink-to-teacher).")
    p.add_argument("--data_dir", type=Path, default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data"))
    p.add_argument("--ensemble_pred_dir", type=Path, default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction/ensemble_lungmasked"))
    p.add_argument("--output_dir", type=Path, default=Path("./results/output_mtcl_single"))
    p.add_argument("--train_split", type=Path, default=Path("./data/splits/train_list.txt"))
    p.add_argument("--val_split", type=Path, default=None)
    p.add_argument("--val_fraction", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_alpha", type=float, default=0.99)

    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_dice", type=float, default=1.0)
    p.add_argument("--lambda_c", type=float, default=0.3)

    p.add_argument("--lambda_cl_max", type=float, default=0.5)
    p.add_argument("--lambda_cl_warmup_iters", type=int, default=4000)

    p.add_argument("--frangi_eps", type=float, default=0.02, help="Small weight for Frangi added to CT.")
    p.add_argument("--tau_ssdm", type=float, default=0.8, help="SSDM shrink factor.")
    p.add_argument("--gamma_class", type=float, default=0.5)

    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--eval_patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--eval_batch_size", type=int, default=8)

    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--refine_every", type=int, default=1, help="Refine pseudo labels every N epochs (epoch-end).")
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument(
        "--init_checkpoint",
        type=Path,
        default=Path(
            "/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_ensemble/checkpoints/model_9/best.pth"
        ),
        help="Pretrained init checkpoint (e.g., from train_ensemble.py).",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=Path,
        default=None,
        help="Resume from a saved MTCL checkpoint like '.../checkpoints/epoch_012.pth'.",
    )
    p.add_argument("--student_noise_std", type=float, default=0.03)
    p.add_argument("--teacher_noise_std", type=float, default=0.01)
    p.add_argument("--save_checkpoints", action="store_true", default=True)

    args = p.parse_args()
    return MTCLConfig(
        data_dir=args.data_dir,
        ensemble_pred_dir=args.ensemble_pred_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        ema_alpha=args.ema_alpha,
        lambda_ce=args.lambda_ce,
        lambda_dice=args.lambda_dice,
        lambda_c=args.lambda_c,
        lambda_cl_max=args.lambda_cl_max,
        lambda_cl_warmup_iters=args.lambda_cl_warmup_iters,
        frangi_eps=args.frangi_eps,
        tau_ssdm=args.tau_ssdm,
        gamma_class=args.gamma_class,
        patch_size=tuple(args.patch_size),
        eval_patch_size=tuple(args.eval_patch_size),
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        refine_every=args.refine_every,
        device=args.device,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        student_noise_std=args.student_noise_std,
        teacher_noise_std=args.teacher_noise_std,
        save_checkpoints=args.save_checkpoints,
    )


def prepare_splits(cfg: MTCLConfig) -> Tuple[List[str], List[str]]:
    train_ids_full = read_id_list(cfg.train_split)
    if cfg.val_split is not None and cfg.val_split.exists():
        val_ids = read_id_list(cfg.val_split)
        train_ids = [cid for cid in train_ids_full if cid not in set(val_ids)]
        return train_ids, val_ids

    rng = random.Random(0)
    ids = train_ids_full.copy()
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * cfg.val_fraction))
    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    return train_ids, val_ids


def train_mtcl(cfg: MTCLConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = prepare_splits(cfg)

    # class weights from GT labels (train set)
    label_paths = [find_matching_file(cfg.data_dir / "labels", cid) for cid in train_ids]
    label_paths = [p for p in label_paths if p is not None]
    class_weights = compute_class_weights(label_paths, cfg.gamma_class).to(device)

    # build student/teacher
    student = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
    teacher = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    start_epoch = 1
    global_step = 0
    if cfg.resume_checkpoint is not None:
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Resume checkpoint at {cfg.resume_checkpoint} does not look like a dict.")

        if isinstance(ckpt.get("student"), dict):
            student.load_state_dict(_strip_module_prefix(ckpt["student"]), strict=True)
        else:
            load_student_checkpoint(student, cfg.resume_checkpoint, device)

        if isinstance(ckpt.get("teacher"), dict):
            teacher.load_state_dict(_strip_module_prefix(ckpt["teacher"]), strict=True)
        else:
            teacher.load_state_dict(student.state_dict())

        if isinstance(ckpt.get("optimizer"), dict):
            optimizer.load_state_dict(ckpt["optimizer"])
            # move optimizer state tensors to the right device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[Resume] Loaded {cfg.resume_checkpoint} (next epoch={start_epoch}, global_step={global_step})")
        if start_epoch > cfg.num_epochs:
            print(f"[Resume] start_epoch={start_epoch} > num_epochs={cfg.num_epochs}; nothing to do.")
            return
    else:
        load_student_checkpoint(student, cfg.init_checkpoint, device)
        teacher.load_state_dict(student.state_dict())

    # init pseudo labels: ensemble mean probs
    p_hat_map = load_ensemble_prob_map(train_ids, cfg.ensemble_pred_dir)
    pseudo_map = {cid: p_hat_map[cid].copy() for cid in train_ids}  # y_tilde init
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        # train dataset (uses current pseudo_map)
        train_ds = VesselPatchDataset(cfg.data_dir, train_ids, cfg.patch_size, pseudo_prob_map=pseudo_map)
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

        metrics, global_step = train_epoch(student, teacher, optimizer, loader, class_weights, cfg, device, global_step)

        # refine pseudo labels at epoch end
        refine_stats = {}
        if cfg.refine_every > 0 and (epoch % cfg.refine_every == 0):
            teacher_probs = generate_teacher_probs(
                teacher, train_ids, cfg.data_dir, cfg.eval_patch_size, device,
                frangi_eps=cfg.frangi_eps, patch_batch_size=cfg.eval_batch_size
            )
            refine_stats = refine_pseudo_map_inplace(pseudo_map, p_hat_map, teacher_probs, tau=cfg.tau_ssdm)

        # optional val dice
        val_dice = evaluate_dice(teacher, val_ids, cfg.data_dir, cfg.eval_patch_size, device, frangi_eps=cfg.frangi_eps) if val_ids else 0.0

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={metrics['loss_total']:.4f} gt={metrics['loss_gt']:.4f} "
            f"pseudo={metrics['loss_pseudo']:.4f} cons={metrics['loss_cons']:.4f} "
            f"lam_cl={metrics['lambda_cl']:.3f} "
            f"noisy_frac={refine_stats.get('avg_noisy_frac', 0.0):.4f} "
            f"val_dice={val_dice:.4f}"
        )

        if cfg.save_checkpoints:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "refine_stats": refine_stats,
                    "config": cfg.__dict__,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pth",
            )

    # final
    if cfg.save_checkpoints:
        torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(), "config": cfg.__dict__}, ckpt_dir / "final.pth")
    print("[Done]")


def main():
    cfg = parse_args()
    train_mtcl(cfg)


if __name__ == "__main__":
    main()
