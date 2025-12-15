"""Fine-tune a saved MTCL leader model on the full training set with Dice + BCE.
python model_script/train/train_mtcl.py \
  --data_dir ./data \
  --train_split ./data/splits/train_list.txt \
  --output_dir ./results/output_mtcl_refit \
  --epochs 50
  --checkpoint /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_mtcl_single/checkpoints/epoch_012.pth

"""

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
from model_script.train import mtcl as mtcl_train
from utils.losses import DiceLoss


def build_loader(data_dir: Path, ids: list[str], patch_size, batch_size: int, num_workers: int) -> DataLoader:
    ds = mtcl_train.VesselPatchDataset(
        data_root=data_dir,
        ids=ids,
        patch_size=patch_size,
        source="train",
        pseudo_map=None,
        use_soft_pseudo=True,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_epoch(model, loader, optimizer, dice_loss_fn, bce_loss_fn, device, lambda_dice: float, lambda_bce: float):
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        imgs = batch["image"].to(device)
        targets = batch["target"].to(device)
        # Use only foreground channel for binary Dice/BCE losses
        targets_fg = targets[:, 1:2]

        logits = model(imgs)
        fg_logits = logits[:, 1:2] if logits.shape[1] > 1 else logits

        dice = dice_loss_fn(fg_logits, targets_fg)
        bce = bce_loss_fn(fg_logits, targets_fg)
        loss = lambda_dice * dice + lambda_bce * bce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an MTCL leader checkpoint on the full training set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=Path("./data"), help="Root containing images/ and labels/")
    parser.add_argument("--train_split", type=Path, default=Path("./data/splits/train_list.txt"), help="Training ID list")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"MTCL checkpoint to start from. If omitted, the latest under {DEFAULT_MTCL_CKPT_DIR} is used.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./results/output_mtcl_refit"), help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for patches")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96], help="Training patch size (D H W)")
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
        try:
            ckpt_path = find_latest_mtcl_checkpoint(DEFAULT_MTCL_CKPT_DIR)
            print(f"[Auto] Using latest MTCL checkpoint: {ckpt_path}")
        except FileNotFoundError as exc:
            raise SystemExit(
                f"No checkpoint provided and none found under {DEFAULT_MTCL_CKPT_DIR}. "
                f"Pass --checkpoint to specify a file."
            ) from exc

    model = load_mtcl_model(ckpt_path, device)
    model.train()

    train_ids = mtcl_train.read_id_list(args.train_split)

    loader = build_loader(args.data_dir, train_ids, tuple(args.patch_size), args.batch_size, args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dice_loss_fn = DiceLoss(sigmoid=True)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = args.output_dir / "leader_refit_best.pth"

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model,
            loader,
            optimizer,
            dice_loss_fn,
            bce_loss_fn,
            device,
            lambda_dice=args.lambda_dice,
            lambda_bce=args.lambda_bce,
        )
        print(f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            },
            args.output_dir / "leader_refit_last.pth",
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                best_path,
            )
            print(f"  Saved new best to {best_path}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
