import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

from utils.unet_model import UNet3D
from utils.dataset import get_train_val_dataloaders
from utils.losses import (
    CombinedLoss,
    DiceLoss,
    calculate_cldice_score,
    calculate_dice_score,
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_cldice = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice_score = calculate_dice_score(outputs, labels)
        cldice_score = calculate_cldice_score(outputs, labels)
        running_loss += loss.item()
        running_dice += dice_score
        running_cldice += cldice_score
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice_score:.4f}',
            'clDice': f'{cldice_score:.4f}'
        })
    
    avg_loss = running_loss / len(train_loader)
    avg_dice = running_dice / len(train_loader)
    avg_cldice = running_cldice / len(train_loader)
    
    return avg_loss, avg_dice, avg_cldice


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_cldice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            dice_score = calculate_dice_score(outputs, labels)
            cldice_score = calculate_cldice_score(outputs, labels)
            running_loss += loss.item()
            running_dice += dice_score
            running_cldice += cldice_score
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_score:.4f}',
                'clDice': f'{cldice_score:.4f}'
            })
    
    avg_loss = running_loss / len(val_loader)
    avg_dice = running_dice / len(val_loader)
    avg_cldice = running_cldice / len(val_loader)
    
    return avg_loss, avg_dice, avg_cldice


def train_unet(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directories
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint to resume from
    start_epoch = 1
    resume_checkpoint_path = None
    if args.resume:
        latest_checkpoint = checkpoint_dir / 'latest.pth'
        if latest_checkpoint.exists():
            resume_checkpoint_path = latest_checkpoint
            print(f'Found checkpoint to resume from: {resume_checkpoint_path}')
        else:
            print('No checkpoint found, starting from scratch')
    
    # Create dataloaders
    print('Loading data...')
    if args.train_list:
        print(f'  Using train list: {args.train_list}')
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        train_list_path=args.train_list
    )
    
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Create model
    print('Creating model...')
    model = UNet3D(n_channels=1, n_classes=1, trilinear=True)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params:,}')
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Resume from checkpoint if exists
    best_dice = 0.0
    best_cldice = 0.0
    if resume_checkpoint_path is not None:
        print(f'Loading checkpoint from {resume_checkpoint_path}')
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', checkpoint['val_dice'])
        best_cldice = checkpoint.get('best_cldice', checkpoint['val_cldice'])
        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'Best Dice so far: {best_dice:.4f}, Best clDice: {best_cldice:.4f}')
        
        if start_epoch > args.epochs:
            print(f'Already completed training ({checkpoint["epoch"]} >= {args.epochs} epochs)')
            print(f'Final Best Dice: {best_dice:.4f}, Best clDice: {best_cldice:.4f}')
            return
    
    # Tensorboard writer
    log_dir = Path(args.output_dir) / 'logs' / datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print(f'\nStarting training from epoch {start_epoch} to {args.epochs}...\n')
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_dice, train_cldice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_dice, val_cldice = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('clDice/train', train_cldice, epoch)
        writer.add_scalar('clDice/val', val_cldice, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train clDice: {train_cldice:.4f}')
        print(f'  Val Loss:   {val_loss:.4f}, Val Dice:   {val_dice:.4f}, Val clDice:   {val_cldice:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'train_cldice': train_cldice,
            'val_cldice': val_cldice,
            'best_dice': best_dice,
            'best_cldice': best_cldice,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint (based on Dice score)
        if val_dice > best_dice:
            best_dice = val_dice
            best_cldice = val_cldice
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f'  >>> New best model saved! Dice: {best_dice:.4f}, clDice: {best_cldice:.4f}\n')
        
        # Save periodic checkpoints
        if epoch % args.save_frequency == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    print(f'\nTraining completed! Best validation Dice: {best_dice:.4f}, clDice: {best_cldice:.4f}')
    print(f'Checkpoints saved to: {checkpoint_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Train 3D U-Net for medical image segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./results/single_model',
                        help='Path to output directory')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Patch size (D H W)')
    parser.add_argument('--train_list', type=str, default=None,
                        help='Optional text file containing image ids for training/validation split')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint if available')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for data loading')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    train_unet(args)


if __name__ == '__main__':
    main()
