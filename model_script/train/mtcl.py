"""Mean-Teacher + Confident Learning (MTCL) trainer for 3D U-Net.

This module wires the math in the user notes into a runnable trainer. It exposes
small helpers (`cl_est`, `refine_labels`, `mtcl_step`) so the flow mirrors the
existing training scripts in this repo.

usage:
python model_script/train/mtcl.py \
  --checkpoint_root /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_ensemble/checkpoints \
  --teacher_id 0 \
  --students all \
  --base_output_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_mtcl \
  --data_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data \
  --epochs 30 --batch_size 4 --refine_mode fs --fs_phi 0.8 \
  --beta 5.0 --lambda_max 0.1 --device cuda:0 --resume

"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import get_train_val_dataloaders
from utils.losses import CombinedLoss, calculate_cldice_score, calculate_dice_score
from utils.unet_model import UNet3D

RefineMode = Literal["hard", "fs", "uds"]


def _stack_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return probabilities for classes {0,1} stacked on channel dim."""
    p1 = torch.sigmoid(logits)
    p0 = 1.0 - p1
    return torch.cat([p0, p1], dim=1)


def _simple_augment(images: torch.Tensor, noise_std: float, flip_prob: float) -> torch.Tensor:
    """Lightweight g(X): random flips and Gaussian noise."""
    out = images
    if flip_prob > 0:
        for dim in (2, 3, 4):  # D, H, W axes (after channel)
            if torch.rand(1, device=images.device) < flip_prob:
                out = torch.flip(out, dims=(dim,))
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


@dataclass
class MTCLConfig:
    beta: float = 5.0
    lambda_max: float = 0.1
    t_max: int | None = None  # filled from epochs * steps if not provided
    refine_mode: RefineMode = "fs"
    fs_phi: float = 0.8
    mc_dropout_samples: int = 4
    use_original_labels: bool = False
    original_label_weight: float = 1.0
    noise_std: float = 0.05
    flip_prob: float = 0.5
    dice_weight: float = 0.5
    bce_weight: float = 0.5


class MTCL:
    """Trainer that refines labels with CL-Est + PBC and aligns student to teacher."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler=None,
        device: torch.device | str = "cuda",
        config: MTCLConfig | None = None,
        augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.config = config or MTCLConfig()
        self.criterion_sup = CombinedLoss(
            dice_weight=self.config.dice_weight, bce_weight=self.config.bce_weight
        )
        self.consistency_fn = nn.MSELoss()
        self.augment_fn = augment_fn
        self.global_step = 0
        self.t_max = self.config.t_max

        # Teacher is frozen (fixed third-party judge)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    # ----------------------- CL-Est + PBC helpers ----------------------- #
    def cl_est(self, labels: torch.Tensor, teacher_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint distribution estimate Q_hat and pixel-level error map X_err.

        Args:
            labels: [B,1,D,H,W] binary labels (0/1)
            teacher_probs: [B,2,D,H,W] teacher probs for classes {0,1}
        Returns:
            Q_hat: [2,2] joint estimate
            error_map: [B,1,D,H,W] bool tensor marking suspected wrong labels
        """
        labels_flat = labels.view(-1).long()
        probs_flat = teacher_probs.permute(0, 2, 3, 4, 1).reshape(-1, 2)
        device = labels.device
        n = labels_flat.numel()

        thresholds = []
        class_counts = torch.zeros(2, device=device, dtype=teacher_probs.dtype)
        for cls in (0, 1):
            cls_mask = labels_flat == cls
            class_counts[cls] = cls_mask.sum()
            if cls_mask.any():
                thresholds.append(probs_flat[cls_mask, cls].mean())
            else:
                thresholds.append(torch.tensor(0.5, device=device, dtype=teacher_probs.dtype))
        thresholds_t = torch.stack(thresholds)  # [2]

        eligible = probs_flat >= thresholds_t  # [n,2], broadcast on dim 0
        eligible_any = eligible.any(dim=1)

        masked_probs = probs_flat.clone()
        masked_probs[~eligible] = -1.0
        argmax_cls = masked_probs.argmax(dim=1)  # [n]

        C = torch.zeros((2, 2), device=device, dtype=teacher_probs.dtype)
        for i in (0, 1):
            obs_mask = labels_flat == i
            obs_and_valid = obs_mask & eligible_any
            for j in (0, 1):
                C[i, j] = (obs_and_valid & (argmax_cls == j)).sum()

        C_tilde = torch.zeros_like(C)
        for i in (0, 1):
            row_sum = C[i].sum()
            if row_sum > 0:
                C_tilde[i] = C[i] / row_sum * class_counts[i]

        denom = C_tilde.sum()
        Q_hat = C_tilde / denom if denom > 0 else C_tilde

        # r_i and m_i (rounded) for prune-by-class
        r0 = Q_hat[0, 1]
        r1 = Q_hat[1, 0]
        m0 = int(torch.round(torch.tensor(n, device=device) * r0).item())
        m1 = int(torch.round(torch.tensor(n, device=device) * r1).item())
        m0 = min(m0, int(class_counts[0].item()))
        m1 = min(m1, int(class_counts[1].item()))

        error_flat = torch.zeros(n, device=device, dtype=torch.bool)
        for cls, m_val in ((0, m0), (1, m1)):
            if m_val <= 0:
                continue
            cls_indices = torch.nonzero(labels_flat == cls, as_tuple=False).squeeze(1)
            if cls_indices.numel() == 0:
                continue
            self_conf = probs_flat[cls_indices, cls]
            if m_val >= self_conf.numel():
                chosen = cls_indices
            else:
                _, rel_idx = torch.topk(self_conf, k=m_val, largest=False)
                chosen = cls_indices[rel_idx]
            error_flat[chosen] = True

        error_map = error_flat.view_as(labels)
        return Q_hat, error_map

    # ----------------------- Label refinement ----------------------- #
    def _uncertainty_map(self, images: torch.Tensor) -> torch.Tensor:
        """MC-dropout uncertainty normalized to [0,1]."""
        preds = []
        dropout_layers = [m for m in self.teacher.modules() if isinstance(m, nn.Dropout)]
        prev_states = [m.training for m in dropout_layers]
        for m in dropout_layers:
            m.train()
        try:
            with torch.no_grad():
                for _ in range(self.config.mc_dropout_samples):
                    logits = self.teacher(images)
                    preds.append(_stack_probs_from_logits(logits))
        finally:
            for layer, state in zip(dropout_layers, prev_states):
                layer.train(state)

        if not preds:
            # No dropout layers present; uncertainty is zero everywhere.
            return torch.zeros_like(images)

        probs_stack = torch.stack(preds, dim=0)  # [T,B,2,D,H,W]
        mean_probs = probs_stack.mean(dim=0)  # [B,2,D,H,W]
        entropy = -torch.sum(mean_probs * torch.clamp(mean_probs, min=1e-8).log(), dim=1, keepdim=True)
        return torch.clamp(entropy / math.log(2.0), 0.0, 1.0)

    def refine_labels(
        self,
        labels: torch.Tensor,
        error_map: torch.Tensor,
        *,
        mode: RefineMode | None = None,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply Hard / Fixed Smooth / UDS refinement to labels using X_err.

        Args:
            labels: [B,1,D,H,W] original labels
            error_map: [B,1,D,H,W] bool/int mask from CL-Est + PBC
            mode: override default refine mode if provided
            images: needed for UDS (MC-dropout)
        """
        mode = mode or self.config.refine_mode
        sign = torch.where(labels > 0.5, -1.0, 1.0)  # (-1)^Y
        err = error_map.float()

        if mode == "hard":
            refined = labels + err * sign
        elif mode == "fs":
            refined = labels + err * sign * self.config.fs_phi
        elif mode == "uds":
            if images is None:
                raise ValueError("images must be provided for UDS refinement")
            u_hat = self._uncertainty_map(images)
            refined = labels + err * sign * (1.0 - u_hat)
        else:
            raise ValueError(f"Unsupported refine mode: {mode}")

        return torch.clamp(refined, 0.0, 1.0)

    # ----------------------- Training step / epoch ----------------------- #
    def _lambda(self) -> float:
        if self.t_max is None or self.t_max <= 0:
            return self.config.lambda_max
        progress = min(self.global_step / float(self.t_max), 1.0)
        return float(self.config.lambda_max * math.exp(-5.0 * (1.0 - progress) ** 2))

    def mtcl_step(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """One MTCL update: teacher -> CL-Est -> refine -> student loss."""
        self.global_step += 1
        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            teacher_logits = self.teacher(images)
            teacher_probs = _stack_probs_from_logits(teacher_logits)
            Q_hat, error_map = self.cl_est(labels, teacher_probs)

        refined_labels = self.refine_labels(labels, error_map, images=images)

        if self.augment_fn:
            student_in = self.augment_fn(images)
        else:
            student_in = _simple_augment(images, self.config.noise_std, self.config.flip_prob)

        student_logits = self.student(student_in)
        student_probs = _stack_probs_from_logits(student_logits)

        L_c = self.consistency_fn(student_probs, teacher_probs.detach())
        L_ls = self.criterion_sup(student_logits, refined_labels)

        total_loss = self._lambda() * (L_c + self.config.beta * L_ls)

        if self.config.use_original_labels:
            orig_loss = self.criterion_sup(student_logits, labels)
            total_loss = total_loss + self.config.original_label_weight * orig_loss
        else:
            orig_loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            dice = calculate_dice_score(student_logits, labels)
            cldice = calculate_cldice_score(student_logits, labels)

        return {
            "loss": float(total_loss.item()),
            "consistency": float(L_c.item()),
            "ls": float(L_ls.item()),
            "orig": float(orig_loss.item()),
            "lambda": self._lambda(),
            "dice": dice,
            "cldice": cldice,
            "q00": float(Q_hat[0, 0].item()),
            "q01": float(Q_hat[0, 1].item()),
            "q10": float(Q_hat[1, 0].item()),
            "q11": float(Q_hat[1, 1].item()),
        }

    def train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int | None = None, show_progress: bool = True) -> dict[str, float]:
        """Loop over one epoch of MTCL training."""
        if self.t_max is None and total_epochs is not None:
            self.t_max = total_epochs * len(loader)

        self.student.train()
        iterator = tqdm(loader, desc=f"Epoch {epoch} [MTCL]", leave=False) if show_progress else loader

        agg = {
            "loss": 0.0,
            "consistency": 0.0,
            "ls": 0.0,
            "orig": 0.0,
            "dice": 0.0,
            "cldice": 0.0,
        }

        for images, labels in iterator:
            stats = self.mtcl_step(images, labels)
            for key in agg:
                agg[key] += stats[key]
            if show_progress:
                iterator.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    dice=f"{stats['dice']:.4f}",
                    lam=f"{stats['lambda']:.3f}",
                )

        n = len(loader)
        return {k: v / n for k, v in agg.items()}

    def evaluate(self, loader: DataLoader, show_progress: bool = False, desc: str | None = None) -> dict[str, float]:
        """Eval student on a loader using the original labels."""
        self.student.eval()
        loss_tot = dice_tot = cldice_tot = 0.0
        iterator = tqdm(loader, desc=desc, leave=False) if show_progress and desc else loader
        with torch.no_grad():
            for images, labels in iterator:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.student(images)
                loss = self.criterion_sup(logits, labels)
                loss_tot += loss.item()
                dice_tot += calculate_dice_score(logits, labels)
                cldice_tot += calculate_cldice_score(logits, labels)
        n = len(loader)
        return {
            "loss": loss_tot / n,
            "dice": dice_tot / n,
            "cldice": cldice_tot / n,
        }

    # ----------------------- Utility helpers ----------------------- #
    @staticmethod
    def load_unet_from_checkpoint(ckpt_path: Path | str, device: torch.device | str) -> UNet3D:
        """Load UNet3D weights from a checkpoint that stores model_state_dict."""
        model = UNet3D(n_channels=1, n_classes=1, trilinear=True).to(device)
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        return model


def build_mtcl_from_checkpoints(
    teacher_ckpt: Path | str,
    student_ckpt: Path | str,
    *,
    device: torch.device | str = "cuda",
    learning_rate: float = 1e-4,
    config: MTCLConfig | None = None,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> MTCL:
    """Convenience factory similar to train_ensemble.build_model."""
    cfg = config or MTCLConfig()
    teacher = MTCL.load_unet_from_checkpoint(teacher_ckpt, device)
    student = MTCL.load_unet_from_checkpoint(student_ckpt, device)
    opt_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_cls(student.parameters(), lr=learning_rate, **opt_kwargs)
    return MTCL(teacher, student, optimizer, device=device, config=cfg)

# ----------------------- Checkpoint + runner helpers ----------------------- #
def _resolve_model_ckpt(root: Path | str, model_id: int) -> Path:
    return Path(root) / f"model_{model_id}" / "best.pth"


def save_mtcl_checkpoint(
    ckpt_path: Path,
    trainer: MTCL,
    epoch: int,
    best: Tuple[float, float],
    train_metrics: dict,
    val_metrics: dict,
    extra_meta: dict | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": trainer.global_step,
        "t_max": trainer.t_max,
        "model_state_dict": trainer.student.state_dict(),
        "teacher_state_dict": trainer.teacher.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_dice": best[0],
        "best_cldice": best[1],
        "config": asdict(trainer.config),
        "meta": extra_meta or {},
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)


def load_mtcl_checkpoint(
    trainer: MTCL,
    ckpt_path: Path,
    *,
    scheduler=None,
) -> Tuple[int, float, float]:
    state = torch.load(ckpt_path, map_location=trainer.device)
    trainer.student.load_state_dict(state["model_state_dict"])
    if "teacher_state_dict" in state:
        trainer.teacher.load_state_dict(state["teacher_state_dict"])
        trainer.teacher.eval()
    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    trainer.global_step = state.get("global_step", trainer.global_step)
    trainer.t_max = state.get("t_max", trainer.t_max)
    best_dice = state.get("best_dice", 0.0)
    best_cldice = state.get("best_cldice", 0.0)
    epoch = state.get("epoch", 0)
    return epoch + 1, best_dice, best_cldice


def run_mtcl_training(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    # No numpy random usage here; add if needed for reproducibility.

    ckpt_root = Path(args.checkpoint_root) if args.checkpoint_root else None
    teacher_path = Path(args.teacher_ckpt) if args.teacher_ckpt else None
    if teacher_path is None and ckpt_root is not None and args.teacher_id is not None:
        teacher_path = _resolve_model_ckpt(ckpt_root, args.teacher_id)
    if teacher_path is None:
        raise ValueError("teacher_ckpt or (checkpoint_root + teacher_id) must be provided.")
    student_path = Path(args.student_ckpt) if args.student_ckpt else None
    if student_path is None:
        raise ValueError("student_ckpt must be provided for single-run mode.")
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_path}")

    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        train_list_path=args.train_list,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    cfg = MTCLConfig(
        beta=args.beta,
        lambda_max=args.lambda_max,
        refine_mode=args.refine_mode,
        fs_phi=args.fs_phi,
        mc_dropout_samples=args.mc_dropout_samples,
        use_original_labels=args.use_original_labels,
        original_label_weight=args.original_label_weight,
        noise_std=args.noise_std,
        flip_prob=args.flip_prob,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
    )

    teacher = MTCL.load_unet_from_checkpoint(teacher_path, device)
    student = MTCL.load_unet_from_checkpoint(student_path, device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    trainer = MTCL(
        teacher,
        student,
        optimizer,
        scheduler=scheduler,
        device=device,
        config=cfg,
    )
    trainer.t_max = len(train_loader) * args.epochs

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir = Path(args.output_dir) / "logs"
    writer = SummaryWriter(log_dir)

    start_epoch = 1
    best_dice = 0.0
    best_cldice = 0.0

    if args.resume:
        latest = ckpt_dir / "latest.pth"
        best_path = ckpt_dir / "best.pth"
        ckpt_to_use = latest if latest.exists() else best_path if best_path.exists() else None
        if ckpt_to_use is not None and ckpt_to_use.exists():
            start_epoch, best_dice, best_cldice = load_mtcl_checkpoint(
                trainer, ckpt_to_use, scheduler=scheduler
            )
            print(f"Resumed from {ckpt_to_use} at epoch {start_epoch - 1}")
            if start_epoch > args.epochs:
                print("Target epochs already reached; exiting.")
                return
        else:
            print("Resume requested but no checkpoint found; starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = trainer.train_epoch(
            train_loader, epoch, total_epochs=args.epochs, show_progress=not args.quiet
        )
        val_metrics = trainer.evaluate(
            val_loader,
            show_progress=not args.quiet,
            desc=f"Epoch {epoch} [Val]" if not args.quiet else None,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_cldice = val_metrics["cldice"]
            print(f"New best Dice {best_dice:.4f} | clDice {best_cldice:.4f}")
            save_mtcl_checkpoint(
                ckpt_dir / "best.pth",
                trainer,
                epoch,
                (best_dice, best_cldice),
                train_metrics,
                val_metrics,
                extra_meta={
                    "teacher_ckpt": str(args.teacher_ckpt),
                    "student_init_ckpt": str(args.student_ckpt),
                },
            )

        save_mtcl_checkpoint(
            ckpt_dir / "latest.pth",
            trainer,
            epoch,
            (best_dice, best_cldice),
            train_metrics,
            val_metrics,
            extra_meta={
                "teacher_ckpt": str(args.teacher_ckpt),
                "student_init_ckpt": str(args.student_ckpt),
            },
        )

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Dice/train", train_metrics["dice"], epoch)
        writer.add_scalar("Dice/val", val_metrics["dice"], epoch)
        writer.add_scalar("clDice/train", train_metrics["cldice"], epoch)
        writer.add_scalar("clDice/val", val_metrics["cldice"], epoch)
        writer.add_scalar("lambda", trainer._lambda(), epoch)
        if scheduler is not None:
            writer.add_scalar("LR", trainer.optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss {train_metrics['loss']:.4f} Dice {train_metrics['dice']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} Dice {val_metrics['dice']:.4f}"
        )

    writer.close()
    print("MTCL training complete.")


def _list_model_ids(root: Path) -> list[int]:
    ids = []
    for path in root.glob("model_*"):
        try:
            mid = int(path.name.split("_")[1])
            ids.append(mid)
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def _parse_students_arg(arg: str, root: Path, teacher_id: int | None) -> list[int]:
    if arg.lower() == "all":
        ids = _list_model_ids(root)
        if teacher_id is not None:
            ids = [i for i in ids if i != teacher_id]
        return ids
    parsed: list[int] = []
    for token in arg.split(","):
        tok = token.strip()
        if tok == "":
            continue
        parsed.append(int(tok))
    if teacher_id is not None:
        parsed = [i for i in parsed if i != teacher_id]
    return sorted(set(parsed))


def run_mtcl_over_students(args) -> None:
    """Iterate MTCL over multiple students using a shared teacher."""
    if not args.students:
        run_mtcl_training(args)
        return

    ckpt_root = Path(args.checkpoint_root)
    teacher_id = args.teacher_id if args.teacher_id is not None else 0
    teacher_path = Path(args.teacher_ckpt) if args.teacher_ckpt else _resolve_model_ckpt(ckpt_root, teacher_id)
    student_ids = _parse_students_arg(args.students, ckpt_root, teacher_id)
    if not student_ids:
        print("No student ids resolved from --students; exiting.")
        return

    for sid in student_ids:
        student_path = _resolve_model_ckpt(ckpt_root, sid)
        out_dir = Path(args.base_output_dir) / f"model_{sid}"
        run_args = copy.deepcopy(args)
        run_args.teacher_ckpt = str(teacher_path)
        run_args.student_ckpt = str(student_path)
        run_args.output_dir = str(out_dir)
        # Each run handles its own resume logic under its output_dir
        print(f"\n=== MTCL student model_{sid} (teacher model_{teacher_id}) ===")
        run_mtcl_training(run_args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MTCL (Mean-Teacher + Confident Learning) fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="Path to frozen teacher checkpoint")
    parser.add_argument("--student_ckpt", type=str, default=None, help="Path to initial student checkpoint")
    parser.add_argument("--teacher_id", type=int, default=0, help="Teacher model id when resolving from checkpoint_root")
    parser.add_argument("--students", type=str, default=None, help="Comma list of student ids or 'all' to iterate")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="./results/output_ensemble/checkpoints",
        help="Root dir containing model_{k}/best.pth",
    )
    parser.add_argument("--output_dir", type=str, default="./results/output_mtcl", help="Output directory (single-run)")
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="./results/output_mtcl",
        help="Base output dir when iterating over multiple students",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--train_list", type=str, default=None, help="Optional train list file")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Patch size D H W")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--epochs", type=int, default=50, help="Number of MTCL epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest/best checkpoint if available")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bars")

    # MTCL-specific
    parser.add_argument("--beta", type=float, default=5.0, help="Weight for label-supervised loss inside lambda(t)")
    parser.add_argument("--lambda_max", type=float, default=0.1, help="Maximum ramp-up weight for consistency block")
    parser.add_argument("--refine_mode", type=str, choices=["hard", "fs", "uds"], default="fs", help="Label refinement strategy")
    parser.add_argument("--fs_phi", type=float, default=0.8, help="Phi for Fixed Smooth refinement")
    parser.add_argument("--mc_dropout_samples", type=int, default=4, help="MC-dropout samples for UDS")
    parser.add_argument("--use_original_labels", action="store_true", help="Include original labels alongside refined labels")
    parser.add_argument("--original_label_weight", type=float, default=1.0, help="Weight of original-label loss term")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Stddev for Gaussian noise augmentation")
    parser.add_argument("--flip_prob", type=float, default=0.5, help="Per-axis flip probability for augmentation")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice weight inside CombinedLoss")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE weight inside CombinedLoss")

    return parser.parse_args()


def main():
    args = parse_args()
    run_mtcl_over_students(args)


if __name__ == "__main__":
    main()
