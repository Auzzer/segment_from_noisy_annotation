# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 17:39:34 2025

@author: jkwar
"""

#!/usr/bin/env python3
"""
STAPLE Ensemble Results Analysis
Analyzes STAPLE fusion results and generates visualization plots.

Usage:
    python analyze_staple_results.py
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM")
RESULTS_DIR = PROJECT_ROOT / "data/nnUNet_results/Dataset003_vesselfm_good/preds"
OUTPUT_DIR = PROJECT_ROOT / "results/staple_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Main analysis function."""
    
    # Load metrics
    metrics_path = RESULTS_DIR / "ensemble_seed1-5" / "metrics.json"
    with open(metrics_path) as f:
        data = json.load(f)

    # Extract per-case Dice scores from metric_per_case
    scores = {}
    if 'metric_per_case' in data:
        for case_entry in data['metric_per_case']:
            pred_file = case_entry.get('prediction_file', '')
            case_name = Path(pred_file).stem if pred_file else None
            
            if 'metrics' in case_entry and case_name:
                metrics = case_entry['metrics']
                # Average Dice across both channels (1 and 2)
                dice_values = []
                for channel in ['1', '2']:
                    if channel in metrics and 'Dice' in metrics[channel]:
                        dice_values.append(metrics[channel]['Dice'])
                
                if dice_values:
                    scores[case_name] = np.mean(dice_values)

    # Get overall foreground mean
    foreground_dice = data.get('foreground_mean', {}).get('Dice', 0)

    # Calculate statistics
    score_values = list(scores.values())
    mean_dice = np.mean(score_values)
    std_dice = np.std(score_values)
    median_dice = np.median(score_values)
    min_dice = np.min(score_values)
    max_dice = np.max(score_values)

    # Print results
    print("="*60)
    print("STAPLE ENSEMBLE RESULTS")
    print("="*60)
    print(f"\nOfficial Foreground Mean: {foreground_dice:.4f}")
    print(f"\nNumber of cases:    {len(scores)}")
    print(f"Mean Dice:          {mean_dice:.4f}")
    print(f"Std Dev:            {std_dice:.4f}")
    print(f"Median Dice:        {median_dice:.4f}")
    print(f"Min Dice:           {min_dice:.4f}")
    print(f"Max Dice:           {max_dice:.4f}")

    print("\nPer-case Dice scores (sorted):")
    print("-" * 40)
    for case_name in sorted(scores.keys()):
        print(f"  {case_name:15s}: {scores[case_name]:.4f}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(score_values, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(mean_dice, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_dice:.4f}')
    ax.axvline(median_dice, color='orange', linestyle=':', linewidth=2, 
               label=f'Median: {median_dice:.4f}')
    ax.set_xlabel('Dice Score', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.set_title('STAPLE Ensemble - Dice Score Distribution (17 cases)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0.70, 0.95)
    plt.tight_layout()
    
    hist_path = OUTPUT_DIR / "staple_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Histogram saved to: {hist_path}")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    case_names = sorted(scores.keys())
    case_scores = [scores[c] for c in case_names]
    x = np.arange(len(case_names))
    bars = ax.bar(x, case_scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(mean_dice, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_dice:.4f}')

    # Color bars below mean in orange
    for bar, score in zip(bars, case_scores):
        if score < mean_dice:
            bar.set_color('orange')

    ax.set_xlabel('Case', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('STAPLE Ensemble - Per-Case Dice Scores', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('case_', '') for c in case_names], 
                        rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0.70, 0.95)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    bar_path = OUTPUT_DIR / "staple_percase_bars.png"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Bar chart saved to: {bar_path}")

    # Save summary statistics
    summary_path = OUTPUT_DIR / "summary_statistics.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STAPLE ENSEMBLE ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Official Foreground Mean Dice: {foreground_dice:.4f}\n\n")
        f.write("Per-Case Statistics:\n")
        f.write("-"*60 + "\n")
        f.write(f"Number of cases: {len(scores)}\n")
        f.write(f"Mean Dice:       {mean_dice:.4f}\n")
        f.write(f"Std Dev:         {std_dice:.4f}\n")
        f.write(f"Median:          {median_dice:.4f}\n")
        f.write(f"Min:             {min_dice:.4f}\n")
        f.write(f"Max:             {max_dice:.4f}\n")
        f.write(f"Range:           {max_dice - min_dice:.4f}\n\n")
        f.write("Per-Case Results:\n")
        f.write("-"*60 + "\n")
        for case_name in sorted(scores.keys()):
            f.write(f"{case_name:15s}: {scores[case_name]:.4f}\n")

    print(f"[SUCCESS] Summary saved to: {summary_path}")

    # Print key results for report
    print("\n" + "="*60)
    print("KEY RESULTS FOR REPORT:")
    print("="*60)
    print(f"STAPLE Mean Dice:   {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Range:              {min_dice:.4f} - {max_dice:.4f}")
    print(f"Official (nnUNet):  {foreground_dice:.4f}")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()