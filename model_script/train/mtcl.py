"""
Mean-Teacher with Frangi-guided consistency and pseudo-HQ labels.

Implements the requested MTCL process:
- 10 student/teacher U-Net pairs with EMA teachers
- student view: photometric aug on normalized CT
- teacher view: Frangi-weighted CT + photometric aug (no geometry change)
- HQ supervision on train split, pseudo-HQ supervision on held-out val split
- leader selection on train HQ, pseudo label refresh on val each epoch
- weighted CE + weighted Dice + consistency loss
Training/validation history is stored under output_dir/history.json.

python model_script/train/mtcl.py 
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

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


def split_train_val_ids(train_ids: List[str], val_fraction: float, seed: int = 0) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = train_ids.copy()
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * val_fraction))
    val_ids = ids[:val_count]
    new_train_ids = ids[val_count:]
    if not new_train_ids:
        # If dataset is extremely small, keep at least one train case
        new_train_ids = ids[1:]
        val_ids = ids[:1]
    return new_train_ids, val_ids


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
            np.pad(
                a,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
            )
            for a in arrays
        ]
        d, h, w = arrays[0].shape
    d0 = random.randint(0, d - pd) if d > pd else 0
    h0 = random.randint(0, h - ph) if h > ph else 0
    w0 = random.randint(0, w - pw) if w > pw else 0
    slices = (slice(d0, d0 + pd), slice(h0, h0 + ph), slice(w0, w0 + pw))
    return [a[slices] for a in arrays]


# ------------------------------- Dataset ----------------------------------- #
class VesselPatchDataset(Dataset):
    """
    Loads CT, HQ labels, Frangi priors, and optional pseudo-HQ labels.
    Returns aligned random patches with consistent geometry.
    """

    def __init__(
        self,
        data_root: Path,
        ids: Iterable[str],
        patch_size: Tuple[int, int, int] | None,
        *,
        source: str,
        pseudo_root: Path | None = None,
        use_soft_pseudo: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.ids = [_normalize_image_id(i) for i in ids]
        self.patch_size = patch_size
        self.source = source
        self.pseudo_root = pseudo_root
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
        if self.pseudo_root is not None:
            pseudo_path = find_matching_file(self.pseudo_root, cid)
            if pseudo_path and pseudo_path.exists():
                pseudo, _ = load_nifti_array(pseudo_path)

        if self.patch_size is not None:
            arrays = [image, label, frangi]
            if pseudo is not None:
                arrays.append(pseudo)
            cropped = random_crop(arrays, self.patch_size)
            image, label, frangi = cropped[:3]
            if pseudo is not None:
                pseudo = cropped[3]

        image = normalize_ct(image)
        frangi = np.clip(frangi, 0.0, 1.0).astype(np.float32)
        label = (label > 0.5).astype(np.float32)
        label_onehot = np.stack([1.0 - label, label], axis=0).astype(np.float32)

        image_t = torch.from_numpy(image[None])
        label_t = torch.from_numpy(label_onehot)
        frangi_t = torch.from_numpy(frangi[None])

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
                pseudo = (pseudo > 0.5).astype(np.float32)
                pseudo = np.stack([1.0 - pseudo, pseudo], axis=0)
            elif pseudo.ndim == 4 and pseudo.shape[0] == 1:
                pseudo = np.concatenate([1.0 - pseudo, pseudo], axis=0)
            sample["target"] = torch.from_numpy(pseudo.astype(np.float32))
            sample["has_pseudo"] = True
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


def vessel_dice(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice on vessel channel only (channel 1), using argmax target if one-hot is provided.
    """
    if targets.shape[1] > 1:
        tgt = targets.argmax(dim=1)
    else:
        tgt = targets.squeeze(1).long()
    pred = (probs[:, 1] >= 0.5).float()
    tgt_bin = (tgt == 1).float()
    inter = (pred * tgt_bin).sum()
    denom = pred.sum() + tgt_bin.sum()
    return (2 * inter + eps) / (denom + eps)


# --------------------------- Augmentations --------------------------------- #
def photometric_augment(x: torch.Tensor, brightness: float, contrast: float, noise_std: float) -> torch.Tensor:
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
    # Center Frangi, then modulate CT intensity.
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
    # Special-case 1-channel heads -> 2-channel heads (copy vessel logit into channel 1, zero bg)
    head_w = "outc.conv.weight"
    head_b = "outc.conv.bias"
    mapped_head = False
    if (
        head_w in state
        and head_b in state
        and head_w in model_state
        and head_b in model_state
        and state[head_w].shape[0] == 1
        and model_state[head_w].shape[0] == 2
    ):
        new_w = model_state[head_w].clone()
        new_b = model_state[head_b].clone()
        new_w.zero_()
        new_b.zero_()
        new_w[1] = state[head_w][0]
        new_b[1] = state[head_b][0]
        filtered_state[head_w] = new_w
        filtered_state[head_b] = new_b
        mapped_head = True
        print(f"[Init] Expanded 1->2 channel head for {path.name}")
    for k, v in state.items():
        if mapped_head and k in (head_w, head_b):
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            shape_skips.append(k)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if shape_skips:
        print(f"[Init] Skipped {len(shape_skips)} mismatched params (e.g., {shape_skips[:2]}) from {path.name}")
    return missing, unexpected


def build_students_and_teachers(
    num_models: int,
    device: torch.device,
    init_root: Path | None = None,
    init_name: str = "best.pth",
) -> Tuple[List[nn.Module], List[nn.Module]]:
    students: List[nn.Module] = []
    teachers: List[nn.Module] = []
    for idx in range(num_models):
        s = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
        if init_root is not None:
            ckpt_dir = init_root / f"model_{idx}"
            candidates = [ckpt_dir / init_name, ckpt_dir / "latest.pth", ckpt_dir / "best.pth"]
            loaded = False
            for cand in candidates:
                if cand.exists():
                    missing, unexpected = load_student_checkpoint(s, cand, device)
                    if missing or unexpected:
                        print(f"[Init model {idx}] Loaded {cand.name} with missing={len(missing)} unexpected={len(unexpected)}")
                    else:
                        print(f"[Init model {idx}] Loaded {cand.name}")
                    loaded = True
                    break
            if not loaded:
                print(f"[Init model {idx}] No checkpoint found under {ckpt_dir}")
        t = UNet3D(n_channels=1, n_classes=2, trilinear=True).to(device)
        t.load_state_dict(s.state_dict())
        students.append(s)
        teachers.append(t)
    return students, teachers


def update_ema(teacher: nn.Module, student: nn.Module, alpha: float):
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.mul_(alpha).add_(s_p.data, alpha=1.0 - alpha)


def stack_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


# ---------------------------- Inference utils ------------------------------ #
@torch.no_grad()
def sliding_window_predict(model: nn.Module, volume: torch.Tensor, patch_size: Tuple[int, int, int], device: torch.device, overlap: float = 0.5) -> torch.Tensor:
    """
    Sliding window inference with simple averaging. volume: [1,1,D,H,W]
    Returns probs [1,2,D,H,W].
    """
    model.eval()
    _, _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    out_sum = torch.zeros((1, 2, D, H, W), device=device)
    weight = torch.zeros((1, 1, D, H, W), device=device)
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
                patch = volume[:, :, sd0:d1, sh0:h1, sw0:w1].to(device)
                logits = model(patch)
                probs = torch.softmax(logits, dim=1)
                out_sum[:, :, sd0:d1, sh0:h1, sw0:w1] += probs
                weight[:, :, sd0:d1, sh0:h1, sw0:w1] += 1.0
    return out_sum / torch.clamp(weight, min=1.0)


def dice_from_probs(probs: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> float:
    pred = (probs >= 0.5).astype(np.float32)
    label = (label > 0.5).astype(np.float32)
    inter = (pred * label).sum()
    denom = pred.sum() + label.sum()
    return float((2 * inter + eps) / (denom + eps))


# ----------------------------- Training config ----------------------------- #
@dataclass
class MTCLConfig:
    data_dir: Path
    output_dir: Path
    train_split: Path
    val_split: Path | None
    val_fraction: float
    epochs: int
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
    num_models: int
    device: str
    pseudo_soft: bool
    save_checkpoints: bool
    init_checkpoint_root: Path | None
    init_checkpoint_name: str
    pseudo_every: int
    max_train_eval: int | None = None


def parse_args() -> MTCLConfig:
    p = argparse.ArgumentParser(description="MTCL trainer with Frangi-guided consistency.")
    p.add_argument("--data_dir", type=Path, default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data"))
    p.add_argument("--output_dir", type=Path, default=Path("./results/output_mtcl"))
    p.add_argument("--train_split", type=Path, default=Path("./data/splits/train_list.txt"))
    p.add_argument("--val_split", type=Path, default=None, help="Optional path to val split; if missing, a val split is created from the train list.")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of train ids to reserve for val when val_split is not provided or missing.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema_alpha", type=float, default=0.99)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_dice", type=float, default=1.0)
    p.add_argument("--lambda_pl", type=float, default=0.5)
    p.add_argument("--lambda_c", type=float, default=0.3)
    p.add_argument("--beta_frangi", type=float, default=0.5)
    p.add_argument("--gamma_class", type=float, default=0.5)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--eval_patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--num_models", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--pseudo_soft", action="store_true", help="Store soft pseudo-HQ labels (default hard argmax).")
    p.add_argument("--save_checkpoints", action="store_true")
    p.add_argument(
        "--init_checkpoint_root",
        type=Path,
        default=Path("/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_ensemble/checkpoints"),
        help="Root containing model_0...model_9 subdirs to warm start students.",
    )
    p.add_argument("--init_checkpoint_name", type=str, default="best.pth", help="Checkpoint filename inside each model_i (fallback to latest.pth).")
    p.add_argument("--pseudo_every", type=int, default=1, help="Generate pseudo-HQ labels every N epochs (1 = every epoch).")
    p.add_argument("--max_train_eval", type=int, default=None, help="Optional cap on train cases for leader scoring.")
    args = p.parse_args()
    return MTCLConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
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
        num_models=args.num_models,
        device=args.device,
        pseudo_soft=args.pseudo_soft,
        save_checkpoints=args.save_checkpoints,
        init_checkpoint_root=args.init_checkpoint_root,
        init_checkpoint_name=args.init_checkpoint_name,
        pseudo_every=args.pseudo_every,
        max_train_eval=args.max_train_eval,
    )


# --------------------------- Metrics & history ----------------------------- #
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


def save_history(history: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(history, f, indent=2)


def save_checkpoint(students: List[nn.Module], teachers: List[nn.Module], optimizer, epoch: int, out_dir: Path):
    ckpt = {
        "epoch": epoch,
        "students": [m.state_dict() for m in students],
        "teachers": [m.state_dict() for m in teachers],
        "optimizer": optimizer.state_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pth")


# ---------------------------- Training epoch ------------------------------- #
def train_one_epoch(
    students: List[nn.Module],
    teachers: List[nn.Module],
    optimizer,
    loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: MTCLConfig,
    device: torch.device,
) -> dict:
    for s in students:
        s.train()
    for t in teachers:
        t.eval()

    total_loss = 0.0
    hq_loss_sum = 0.0
    ph_loss_sum = 0.0
    cons_loss_sum = 0.0
    dice_hq_w_sum = 0.0
    dice_ph_w_sum = 0.0
    dice_hq_vessel_sum = 0.0
    dice_ph_vessel_sum = 0.0
    n_batches = 0
    n_hq_batches = 0
    n_ph_batches = 0

    class_weights = class_weights.to(device)

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        frangi = batch["frangi"].to(device)
        targets = batch["target"].to(device)
        sources = batch["source"]
        has_pseudo = batch["has_pseudo"]

        x_s = photometric_augment(imgs, brightness=0.1, contrast=0.1, noise_std=0.03)
        x_t = frangi_weighted_view(imgs, frangi, beta=cfg.beta_frangi)
        x_t = photometric_augment(x_t, brightness=0.1, contrast=0.1, noise_std=0.03)

        student_probs = []
        for s in students:
            logits = s(x_s)
            student_probs.append(stack_softmax(logits))
        S = torch.stack(student_probs, dim=0).mean(dim=0)

        teacher_probs = []
        with torch.no_grad():
            for t in teachers:
                logits_t = t(x_t)
                teacher_probs.append(stack_softmax(logits_t))
        T = torch.stack(teacher_probs, dim=0).mean(dim=0).detach()

        train_mask = torch.tensor([src == "train" for src in sources], device=device, dtype=torch.bool)
        val_mask = torch.tensor(
            [(src == "val") and bool(p) for src, p in zip(sources, has_pseudo)],
            device=device,
            dtype=torch.bool,
        )

        loss_hq = torch.tensor(0.0, device=device)
        loss_ph = torch.tensor(0.0, device=device)

        if train_mask.any():
            tgt_hq = targets[train_mask]
            if tgt_hq.shape[1] == 1:
                tgt_hq = to_one_hot(tgt_hq)
            loss_ce = weighted_cross_entropy(S[train_mask], tgt_hq, class_weights)
            loss_dice = weighted_dice_loss(S[train_mask], tgt_hq, class_weights)
            loss_hq = cfg.lambda_ce * loss_ce + cfg.lambda_dice * loss_dice
            dice_batch_w = 1.0 - loss_dice.detach()
            dice_batch_v = vessel_dice(S[train_mask], tgt_hq).detach()
            dice_hq_w_sum += dice_batch_w.item()
            dice_hq_vessel_sum += dice_batch_v.item()
            n_hq_batches += 1

        if val_mask.any():
            tgt_ph = targets[val_mask]
            if tgt_ph.shape[1] == 1:
                tgt_ph = to_one_hot(tgt_ph)
            loss_ce_ph = weighted_cross_entropy(S[val_mask], tgt_ph, class_weights)
            loss_dice_ph = weighted_dice_loss(S[val_mask], tgt_ph, class_weights)
            loss_ph = cfg.lambda_ce * loss_ce_ph + cfg.lambda_dice * loss_dice_ph
            dice_batch_w = 1.0 - loss_dice_ph.detach()
            dice_batch_v = vessel_dice(S[val_mask], tgt_ph).detach()
            dice_ph_w_sum += dice_batch_w.item()
            dice_ph_vessel_sum += dice_batch_v.item()
            n_ph_batches += 1

        loss_cons = F.mse_loss(S, T)
        loss = loss_hq + cfg.lambda_pl * loss_ph + cfg.lambda_c * loss_cons

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for s_model, t_model in zip(students, teachers):
            update_ema(t_model, s_model, cfg.ema_alpha)

        total_loss += loss.item()
        hq_loss_sum += loss_hq.item()
        ph_loss_sum += loss_ph.item()
        cons_loss_sum += loss_cons.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "hq_loss": hq_loss_sum / max(n_batches, 1),
        "ph_loss": ph_loss_sum / max(n_batches, 1),
        "cons_loss": cons_loss_sum / max(n_batches, 1),
        "hq_weighted_dice": dice_hq_w_sum / max(n_hq_batches, 1),
        "ph_weighted_dice": dice_ph_w_sum / max(n_ph_batches, 1),
        "hq_vessel_dice": dice_hq_vessel_sum / max(n_hq_batches, 1),
        "ph_vessel_dice": dice_ph_vessel_sum / max(n_ph_batches, 1),
    }


# ------------------------- Leader selection & PL --------------------------- #
def eval_teacher_on_cases(
    teacher: nn.Module,
    ids: List[str],
    data_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    max_cases: int | None = None,
) -> float:
    scores = []
    for idx, cid in enumerate(ids):
        if max_cases is not None and idx >= max_cases:
            break
        img_path = find_matching_file(data_dir / "images", cid)
        lbl_path = find_matching_file(data_dir / "labels", cid)
        if img_path is None or lbl_path is None:
            continue
        image, _ = load_nifti_array(img_path)
        label, _ = load_nifti_array(lbl_path)
        image = normalize_ct(image)
        vol = torch.from_numpy(image[None, None]).to(device)
        probs = sliding_window_predict(teacher, vol, patch_size, device)[0].cpu().numpy()
        dice = dice_from_probs(probs[1], label)
        scores.append(dice)
    return float(np.mean(scores)) if scores else 0.0


def select_leader(
    teachers: List[nn.Module],
    train_ids: List[str],
    data_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    max_cases: int | None = None,
) -> Tuple[int, List[float]]:
    qualities = []
    for t in teachers:
        q = eval_teacher_on_cases(t, train_ids, data_dir, patch_size, device, max_cases=max_cases)
        qualities.append(q)
    leader = int(np.argmax(qualities))
    return leader, qualities


@torch.no_grad()
def generate_pseudo_labels(
    teacher: nn.Module,
    val_ids: List[str],
    data_dir: Path,
    out_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    soft: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in tqdm(val_ids, desc="Pseudo labels", leave=False):
        img_path = find_matching_file(data_dir / "images", cid)
        if img_path is None:
            continue
        image, affine = load_nifti_array(img_path)
        image = normalize_ct(image)
        vol = torch.from_numpy(image[None, None]).to(device)
        probs = sliding_window_predict(teacher, vol, patch_size, device)[0].cpu().numpy()
        if soft:
            out_arr = probs.astype(np.float32)
        else:
            hard = np.argmax(probs, axis=0).astype(np.uint8)
            out_arr = hard
        nib.save(nib.Nifti1Image(out_arr, affine), out_dir / f"{cid}.nii.gz")


@torch.no_grad()
def evaluate_ensemble(
    teachers: List[nn.Module],
    ids: List[str],
    data_dir: Path,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> float:
    scores = []
    for cid in ids:
        img_path = find_matching_file(data_dir / "images", cid)
        lbl_path = find_matching_file(data_dir / "labels", cid)
        if img_path is None or lbl_path is None:
            continue
        image, _ = load_nifti_array(img_path)
        label, _ = load_nifti_array(lbl_path)
        image = normalize_ct(image)
        vol = torch.from_numpy(image[None, None]).to(device)
        teacher_probs = []
        for t in teachers:
            t.eval()
            probs = sliding_window_predict(t, vol, patch_size, device)[0]
            teacher_probs.append(probs)
        ensemble = torch.stack(teacher_probs, dim=0).mean(dim=0).cpu().numpy()
        dice = dice_from_probs(ensemble[1], label)
        scores.append(dice)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------- Training driver ------------------------------ #
def train_mtcl(cfg: MTCLConfig):
    req = cfg.device.lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(cfg.device)
    elif req.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_root = cfg.output_dir / "pseudo_hq"
    ckpt_dir = cfg.output_dir / "checkpoints"
    history_path = cfg.output_dir / "history.json"

    train_ids_full = read_id_list(cfg.train_split)
    print(f"[Init] Loaded {len(train_ids_full)} ids from {cfg.train_split}")
    val_ids: List[str]
    train_ids: List[str]
    if cfg.val_split is not None and cfg.val_split.exists():
        val_ids = read_id_list(cfg.val_split)
        val_set = set(val_ids)
        train_ids = [cid for cid in train_ids_full if cid not in val_set]
        print(f"[Init] Using provided val split ({len(val_ids)} cases); train set after exclusion: {len(train_ids)} cases")
    else:
        train_ids, val_ids = split_train_val_ids(train_ids_full, cfg.val_fraction, seed=0)
        # Save the derived split for reproducibility
        split_dir = cfg.train_split.parent
        split_dir.mkdir(parents=True, exist_ok=True)
        val_path = cfg.val_split if cfg.val_split is not None else split_dir / "val_from_train.txt"
        with open(val_path, "w") as f:
            for cid in val_ids:
                f.write(f"{cid}\n")
        derived_train_path = split_dir / "train_from_train.txt"
        with open(derived_train_path, "w") as f:
            for cid in train_ids:
                f.write(f"{cid}\n")
        print(f"[Split] Created val split at {val_path} with {len(val_ids)} cases; updated train set has {len(train_ids)} cases.")
    label_paths = [find_matching_file(cfg.data_dir / "labels", cid) for cid in train_ids]
    label_paths = [p for p in label_paths if p is not None]
    class_weights = compute_class_weights(label_paths, gamma=cfg.gamma_class)

    students, teachers = build_students_and_teachers(
        cfg.num_models,
        device,
        init_root=cfg.init_checkpoint_root,
        init_name=cfg.init_checkpoint_name,
    )
    print(f"[Init] Students/teachers ready; checkpoints from {cfg.init_checkpoint_root}")
    params = [p for m in students for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    history: List[dict] = []
    prev_pseudo_dir: Path | None = None

    for epoch in range(1, cfg.epochs + 1):
        # Build loaders (add val pseudo set if available)
        train_ds = VesselPatchDataset(
            cfg.data_dir,
            train_ids,
            cfg.patch_size,
            source="train",
            pseudo_root=None,
        )
        datasets = [train_ds]
        if prev_pseudo_dir is not None and prev_pseudo_dir.exists():
            val_ds = VesselPatchDataset(
                cfg.data_dir,
                val_ids,
                cfg.patch_size,
                source="val",
                pseudo_root=prev_pseudo_dir,
                use_soft_pseudo=cfg.pseudo_soft,
            )
            datasets.append(val_ds)
        mixed_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        train_loader = DataLoader(
            mixed_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        print(f"[Epoch {epoch:02d}] Data ready: train batches={len(train_loader)}; pseudo_dir={'none' if prev_pseudo_dir is None else prev_pseudo_dir}")

        train_metrics = train_one_epoch(
            students,
            teachers,
            optimizer,
            train_loader,
            class_weights,
            cfg,
            device,
        )

        leader_idx, qualities = select_leader(
            teachers,
            train_ids,
            cfg.data_dir,
            cfg.eval_patch_size,
            device,
            max_cases=cfg.max_train_eval,
        )

        if prev_pseudo_dir is None or (epoch % max(cfg.pseudo_every, 1) == 0):
            pseudo_dir = pseudo_root / f"epoch_{epoch:03d}"
            generate_pseudo_labels(
                teachers[leader_idx],
                val_ids,
                cfg.data_dir,
                pseudo_dir,
                cfg.eval_patch_size,
                device,
                soft=cfg.pseudo_soft,
            )
            prev_pseudo_dir = pseudo_dir
            print(f"[Epoch {epoch:02d}] Pseudo labels saved to {pseudo_dir}")
        else:
            print(f"[Epoch {epoch:02d}] Reusing pseudo labels from {prev_pseudo_dir}")

        val_dice = evaluate_ensemble(
            teachers,
            val_ids,
            cfg.data_dir,
            cfg.eval_patch_size,
            device,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "leader_idx": leader_idx,
            "leader_scores": qualities,
            "val_dice": val_dice,
        }
        history.append(record)
        save_history(history, history_path)
        print(f"[Epoch {epoch:02d}] History updated at {history_path}")

        if cfg.save_checkpoints:
            save_checkpoint(students, teachers, optimizer, epoch, ckpt_dir)
            print(f"[Epoch {epoch:02d}] Checkpoint saved to {ckpt_dir}")

        print(
            f"Epoch {epoch:02d} | loss {train_metrics['loss']:.4f} | "
            f"HQ vessel dice {train_metrics['hq_vessel_dice']:.4f} | "
            f"Val Dice {val_dice:.4f} | Leader {leader_idx}"
        )


def main():
    cfg = parse_args()
    train_mtcl(cfg)


if __name__ == "__main__":
    main()
