"""Fine-tune the MTCL leader model with Dice + BCE loss (train_ensemble style)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_script.predict.predict_mtcl_model import (
    DEFAULT_MTCL_CKPT_DIR,
    find_latest_mtcl_checkpoint,
    load_mtcl_model,
)
from utils.dataset import get_train_val_dataloaders  # type: ignore
from utils.losses import DiceLoss, calculate_dice_score


def _foreground_logits(logits: torch.Tensor) -> torch.Tensor:
    """Use foreground channel logits (assumes C=2) or pass-through when C=1."""
    if logits.shape[1] > 1:
        return logits[:, 1:2]
    return logits


def train_epoch(model, loader: DataLoader, optimizer, dice_loss, bce_loss, lambda_dice: float, lambda_bce: float, device):
    model.train()
    loss_tot = 0.0
    dice_tot = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        fg_logits = _foreground_logits(logits)

        dice = dice_loss(fg_logits, labels)
        bce = bce_loss(fg_logits, labels)
        loss = lambda_dice * dice + lambda_bce * bce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tot += loss.item()
        dice_tot += calculate_dice_score(fg_logits, labels)
        n += 1
    n = max(n, 1)
    return {"loss": loss_tot / n, "dice": dice_tot / n}


@torch.no_grad()
def eval_epoch(model, loader: DataLoader, dice_loss, bce_loss, lambda_dice: float, lambda_bce: float, device):
    model.eval()
    loss_tot = 0.0
    dice_tot = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        fg_logits = _foreground_logits(logits)

        dice = dice_loss(fg_logits, labels)
        bce = bce_loss(fg_logits, labels)
        loss = lambda_dice * dice + lambda_bce * bce

        loss_tot += loss.item()
        dice_tot += calculate_dice_score(fg_logits, labels)
        n += 1
    n = max(n, 1)
    return {"loss": loss_tot / n, "dice": dice_tot / n}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MTCL leader model with Dice+BCE (train_ensemble style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=Path("./data"), help="Root containing images/ and labels/")
    parser.add_argument("--train_list", type=Path, default=Path("./data/splits/train_list.txt"), help="Training ID list")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training set used for validation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"MTCL checkpoint to start from (defaults to latest under {DEFAULT_MTCL_CKPT_DIR}).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./results/output_mtcl_leader_bce_dice"), help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for patches")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96], help="Patch size (D H W)")
    parser.add_argument("--lambda_dice", type=float, default=0.5, help="Dice loss weight")
    parser.add_argument("--lambda_bce", type=float, default=0.5, help="BCE loss weight")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_latest_mtcl_checkpoint(DEFAULT_MTCL_CKPT_DIR)
        print(f"[Auto] Using latest MTCL checkpoint: {ckpt_path}")

    # Build model and load weights
    model = load_mtcl_model(ckpt_path, device)
    model.train()

    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        train_list_path=str(args.train_list) if args.train_list else None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dice_loss = DiceLoss(sigmoid=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -float("inf")
    last_path = args.output_dir / "leader_bce_dice_last.pth"
    best_path = args.output_dir / "leader_bce_dice_best.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            dice_loss,
            bce_loss,
            args.lambda_dice,
            args.lambda_bce,
            device,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            dice_loss,
            bce_loss,
            args.lambda_dice,
            args.lambda_bce,
            device,
        )
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} dice {val_metrics['dice']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            last_path,
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"  Saved new best to {best_path}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
