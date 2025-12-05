"""
Train an ensemble of models with configurable process-level parallelism.

Each model runs in its own process (optional limit via --num_parallel) so you can
train multiple models concurrently across one or more GPUs. Checkpoints/logs are
written per-model under the provided output directory.
"""

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.dataset import get_train_val_dataloaders
from utils.losses import (
    CombinedLoss,
    calculate_cldice_score,
    calculate_dice_score,
)
from utils.unet_model import UNet3D


def build_model(device: torch.device, learning_rate: float):
    model = UNet3D(n_channels=1, n_classes=1, trilinear=True).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    return model, opt, sch


def save_checkpoint(
    model_id: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    train_metrics,
    val_metrics,
    best: Tuple[float, float],
    ckpt_dir: Path,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "train_dice": train_metrics["dice"],
        "val_dice": val_metrics["dice"],
        "train_cldice": train_metrics["cldice"],
        "val_cldice": val_metrics["cldice"],
        "best_dice": best[0],
        "best_cldice": best[1],
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_dir / "latest.pth")
    if val_metrics["dice"] >= best[0]:
        torch.save(ckpt, ckpt_dir / "best.pth")


def step_model(model, optimizer, criterion, images, labels):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    dice = calculate_dice_score(outputs, labels)
    cldice = calculate_cldice_score(outputs, labels)
    return loss.item(), dice, cldice


def load_checkpoint_safe(path: Path, device: torch.device, model_id: int):
    """
    Attempt to load a checkpoint, tolerating corruption by returning None instead
    of raising. This lets training continue when a partial/invalid file exists.
    """
    try:
        return torch.load(path, map_location=device)
    except (RuntimeError, EOFError, pickle.UnpicklingError, ValueError) as exc:
        print(f"[Model {model_id}] Checkpoint at {path} is unreadable ({exc}); ignoring it.")
        return None


def evaluate_model(model, criterion, loader, device, desc: str | None = None, show_progress: bool = False):
    model.eval()
    loss_tot = dice_tot = cldice_tot = 0.0
    iterator = tqdm(loader, desc=desc, leave=False) if show_progress and desc else loader
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_tot += loss.item()
            dice_tot += calculate_dice_score(outputs, labels)
            cldice_tot += calculate_cldice_score(outputs, labels)
    n = len(loader)
    return loss_tot / n, dice_tot / n, cldice_tot / n


def run_train_epoch(
    model,
    optimizer,
    criterion,
    loader,
    device,
    model_id: int,
    epoch: int,
    show_progress: bool,
):
    model.train()
    loss_tot = dice_tot = cldice_tot = 0.0
    iterator = tqdm(
        loader,
        desc=f"Model {model_id} Epoch {epoch} [Train]",
        leave=False,
    ) if show_progress else loader

    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        loss, dice, cldice = step_model(model, optimizer, criterion, images, labels)
        loss_tot += loss
        dice_tot += dice
        cldice_tot += cldice

    n = len(loader)
    return {"loss": loss_tot / n, "dice": dice_tot / n, "cldice": cldice_tot / n}


def resolve_devices(args) -> List[str]:
    """
    Determine which devices to use. If --devices is provided it takes priority,
    otherwise --device is used. CPU fallback is automatic when CUDA is missing.
    """
    if args.devices:
        requested = [d.strip() for d in args.devices.split(",") if d.strip()]
    elif args.device:
        requested = [args.device]
    else:
        requested = []

    if not requested:
        requested = ["cuda:0" if torch.cuda.is_available() else "cpu"]

    normalized: List[str] = []
    for entry in requested:
        lower = entry.lower()
        if lower == "cpu":
            normalized.append("cpu")
        elif lower.startswith("cuda:"):
            normalized.append(entry)
        elif lower.isdigit():
            normalized.append(f"cuda:{entry}")
        else:
            normalized.append(entry)

    if not torch.cuda.is_available():
        return ["cpu"]

    return normalized


def train_single_model(model_id: int, args, device_str: str, show_progress: bool = True):
    """
    Train one model. This function is executed in the main process (serial mode)
    or in a child process (parallel mode).
    """
    device = torch.device(device_str if device_str != "cpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[Model {model_id}] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device)
        except RuntimeError as exc:
            print(
                f"[Model {model_id}] Unable to select CUDA device {device_str} "
                f"({exc}); falling back to CPU."
            )
            device = torch.device("cpu")

    # Unique seeds per model so initialization differs across ensemble members
    torch.manual_seed(args.seed + model_id)
    np.random.seed(args.seed + model_id)

    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        train_list_path=args.train_list,
    )
    print(
        f"[Model {model_id}] Using device {device} | "
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    try:
        model, optimizer, scheduler = build_model(device, args.learning_rate)
    except RuntimeError as exc:
        if device.type == "cuda":
            print(
                f"[Model {model_id}] Failed to initialize model on CUDA ({exc}); "
                "retrying on CPU."
            )
            device = torch.device("cpu")
            model, optimizer, scheduler = build_model(device, args.learning_rate)
        else:
            raise
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)

    ckpt_dir = Path(args.output_dir) / "checkpoints" / f"model_{model_id}"
    writer = SummaryWriter(Path(args.output_dir) / "logs" / f"model_{model_id}")

    start_epoch = 1
    best_dice = 0.0
    best_cldice = 0.0

    if args.resume:
        latest = ckpt_dir / "latest.pth"
        best_path = ckpt_dir / "best.pth"
        ckpt = None
        ckpt_source = None

        if latest.exists():
            ckpt = load_checkpoint_safe(latest, device, model_id)
            ckpt_source = "latest"
        if ckpt is None and best_path.exists():
            ckpt = load_checkpoint_safe(best_path, device, model_id)
            ckpt_source = "best"

        if ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_dice = ckpt.get("best_dice", ckpt.get("val_dice", 0.0))
            best_cldice = ckpt.get("best_cldice", ckpt.get("val_cldice", 0.0))
            print(f"[Model {model_id}] Resuming from epoch {start_epoch - 1} ({ckpt_source}.pth)")
            if start_epoch > args.epochs:
                print(f"[Model {model_id}] Target epochs already reached; skipping training.")
                writer.close()
                return
        elif latest.exists() or best_path.exists():
            print(f"[Model {model_id}] No readable checkpoint found in {ckpt_dir}; starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            model_id,
            epoch,
            show_progress=show_progress,
        )
        v_loss, v_dice, v_cldice = evaluate_model(
            model,
            criterion,
            val_loader,
            device,
            desc=f"Model {model_id} Epoch {epoch} [Val]" if show_progress else None,
            show_progress=show_progress,
        )
        val_metrics = {"loss": v_loss, "dice": v_dice, "cldice": v_cldice}
        scheduler.step(v_dice)

        # Update bests before saving
        if v_dice > best_dice:
            best_dice = v_dice
            best_cldice = v_cldice
            print(f"[Model {model_id}] New best Dice {best_dice:.4f}, clDice {best_cldice:.4f}")

        save_checkpoint(
            model_id,
            epoch,
            model,
            optimizer,
            scheduler,
            train_metrics,
            val_metrics,
            (best_dice, best_cldice),
            ckpt_dir,
        )

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Dice/train", train_metrics["dice"], epoch)
        writer.add_scalar("Dice/val", val_metrics["dice"], epoch)
        writer.add_scalar("clDice/train", train_metrics["cldice"], epoch)
        writer.add_scalar("clDice/val", val_metrics["cldice"], epoch)

        print(
            f"[Model {model_id}] Epoch {epoch}/{args.epochs} "
            f"| Train Loss {train_metrics['loss']:.4f} Dice {train_metrics['dice']:.4f} "
            f"| Val Loss {val_metrics['loss']:.4f} Dice {val_metrics['dice']:.4f}"
        )

    writer.close()
    print(f"[Model {model_id}] Training complete.")


def train_ensemble(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    devices = resolve_devices(args)
    unique_cuda = {d for d in devices if d.startswith("cuda")}
    parallel_limit = max(1, args.num_parallel)
    if len(unique_cuda) <= 1 and torch.cuda.device_count() <= 1 and parallel_limit > 1:
        print(
            "Detected a single CUDA device; reducing --num_parallel to 1 to avoid "
            "multi-process CUDA contention. Use --devices or CUDA_VISIBLE_DEVICES to "
            "target multiple GPUs."
        )
        parallel_limit = 1
    if devices == ["cpu"] and parallel_limit > 1:
        print("CPU-only environment detected; proceeding with parallel CPU workers.")
    args.num_parallel = parallel_limit

    device_assignments = {mid: devices[mid % len(devices)] for mid in range(args.num_models)}
    device_msg = ", ".join(f"{mid}->{dev}" for mid, dev in device_assignments.items())
    print(f"Device assignments: {device_msg}")
    print(f"Parallel models: {args.num_parallel} (max per batch)")

    if args.num_parallel <= 1:
        for mid in range(args.num_models):
            train_single_model(mid, args, device_assignments[mid], show_progress=True)
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in the current interpreter
        pass

    active: List[Tuple[int, mp.Process]] = []
    errors: List[Tuple[int, int]] = []

    for mid in range(args.num_models):
        # Respect the parallelism limit
        while len(active) >= args.num_parallel:
            finished_mid, proc = active.pop(0)
            proc.join()
            if proc.exitcode != 0:
                errors.append((finished_mid, proc.exitcode))

        proc = mp.Process(
            target=train_single_model,
            args=(mid, args, device_assignments[mid], False),
            name=f"model-{mid}",
        )
        proc.start()
        active.append((mid, proc))

    # Wait for remaining processes
    for finished_mid, proc in active:
        proc.join()
        if proc.exitcode != 0:
            errors.append((finished_mid, proc.exitcode))

    if errors:
        failed = ", ".join(f"model {mid} (code {code})" for mid, code in errors)
        raise RuntimeError(f"Some models failed during training: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel ensemble training with per-model checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_models", type=int, default=10, help="Number of models to train")
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=1,
        help="Number of models to train simultaneously (processes)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices to target (e.g., '0,1' or 'cuda:0,cuda:1'); "
        "falls back to --device or CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Single device to use when --devices is not provided",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--train_list", type=str, default=None, help="Optional train list")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Patch size D H W")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./results/output_ensemble", help="Output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoints if they are present",
    )

    args = parser.parse_args()
    train_ensemble(args)


if __name__ == "__main__":
    main()
