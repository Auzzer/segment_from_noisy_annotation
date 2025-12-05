#!/usr/bin/env python3
"""
Check training progress for ensemble models.
Shows which models are complete, in progress, or not started.
"""

import argparse
from pathlib import Path
import torch
from datetime import datetime


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def check_model_progress(checkpoint_path, target_epochs):
    """Check progress of a single model"""
    latest_checkpoint = checkpoint_path / 'latest.pth'
    best_checkpoint = checkpoint_path / 'best.pth'
    
    if not latest_checkpoint.exists():
        return {
            'status': 'not_started',
            'epoch': 0,
            'progress': 0.0,
            'val_dice': 0.0,
            'best_dice': 0.0,
        }
    
    try:
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        epoch = checkpoint['epoch']
        val_dice = checkpoint.get('val_dice', 0.0)
        best_dice = checkpoint.get('best_dice', val_dice)
        progress = (epoch / target_epochs) * 100
        
        if epoch >= target_epochs:
            status = 'completed'
        else:
            status = 'in_progress'
        
        return {
            'status': status,
            'epoch': epoch,
            'progress': progress,
            'val_dice': val_dice,
            'best_dice': best_dice,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'epoch': 0,
            'progress': 0.0,
            'val_dice': 0.0,
            'best_dice': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Check ensemble training progress',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output_dir', type=str, default='./results/output_ensemble',
                        help='Output directory containing checkpoints')
    parser.add_argument('--num_models', type=int, default=10,
                        help='Number of models in ensemble')
    parser.add_argument('--target_epochs', type=int, default=100,
                        help='Target number of epochs')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    checkpoint_base = output_dir / 'checkpoints'
    
    if not checkpoint_base.exists():
        print(f"Checkpoint directory not found: {checkpoint_base}")
        print("Have you started training yet?")
        return
    
    print("=" * 80)
    print(f"Ensemble Training Progress Report")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target: {args.target_epochs} epochs per model")
    print("=" * 80)
    print()
    
    # Check each model
    results = []
    for model_id in range(args.num_models):
        checkpoint_path = checkpoint_base / f'model_{model_id}'
        progress = check_model_progress(checkpoint_path, args.target_epochs)
        progress['model_id'] = model_id
        results.append(progress)
    
    # Categorize models
    completed = [r for r in results if r['status'] == 'completed']
    in_progress = [r for r in results if r['status'] == 'in_progress']
    not_started = [r for r in results if r['status'] == 'not_started']
    errors = [r for r in results if r['status'] == 'error']
    
    # Print summary
    print(f"Summary:")
    print(f"   Completed:    {len(completed)}/{args.num_models}")
    print(f"   In Progress:  {len(in_progress)}/{args.num_models}")
    print(f"   Not Started:  {len(not_started)}/{args.num_models}")
    if errors:
        print(f"   Errors:       {len(errors)}/{args.num_models}")
    print()
    
    # Print details for completed models
    if completed:
        print(f"Completed Models ({len(completed)}):")
        for r in completed:
            print(f"   Model {r['model_id']:2d}: {r['epoch']:3d}/{args.target_epochs} epochs | "
                  f"Best Dice: {r['best_dice']:.4f} | Current Dice: {r['val_dice']:.4f}")
        print()
    
    # Print details for in-progress models
    if in_progress:
        print(f"In Progress Models ({len(in_progress)}):")
        for r in in_progress:
            bar_length = 20
            filled = int(bar_length * r['progress'] / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            print(f"   Model {r['model_id']:2d}: [{bar}] {r['progress']:5.1f}% | "
                  f"Epoch {r['epoch']:3d}/{args.target_epochs} | "
                  f"Best Dice: {r['best_dice']:.4f}")
        print()
    
    # Print details for not started models
    if not_started:
        print(f"Not Started Models ({len(not_started)}):")
        for r in not_started:
            print(f"   Model {r['model_id']:2d}: Waiting to start")
        print()
    
    # Print errors
    if errors:
        print(f"Models with Errors ({len(errors)}):")
        for r in errors:
            print(f"   Model {r['model_id']:2d}: {r.get('error', 'Unknown error')}")
        print()
    
    # Calculate overall progress
    total_epochs = sum(r['epoch'] for r in results)
    target_total = args.num_models * args.target_epochs
    overall_progress = (total_epochs / target_total) * 100
    
    print("=" * 80)
    print(f"Overall Progress: {total_epochs}/{target_total} epochs ({overall_progress:.1f}%)")
    
    # Estimate completion
    if in_progress or not_started:
        remaining_epochs = target_total - total_epochs
        print(f"Remaining: {remaining_epochs} epochs")
        print()
        print("To resume training:")
        print("   ./scripts/run_ensemble_training.sh --resume")
    else:
        print()
        print("All models completed!")
        avg_dice = sum(r['best_dice'] for r in completed) / len(completed) if completed else 0
        print(f"Average Best Dice: {avg_dice:.4f}")
        print()
        print("Ready for ensemble prediction!")
        print("   python predict_ensemble.py --input your_image.nii.gz")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
