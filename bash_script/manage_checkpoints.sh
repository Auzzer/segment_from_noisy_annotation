#!/bin/bash
set -euo pipefail

# Checkpoint Management Helper Script
# Utilities for managing training checkpoints and resuming training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/noisy_seg_env/bin/python"

show_usage() {
    echo "========================================"
    echo "Checkpoint Management Helper"
    echo "========================================"
    echo ""
    echo "Usage:"
    echo "  $0 status              - Check training progress for all models"
    echo "  $0 resume              - Resume training from checkpoints"
    echo "  $0 clean               - Remove all checkpoints (start fresh)"
    echo "  $0 clean-model N       - Remove checkpoints for model N"
    echo "  $0 backup              - Backup all checkpoints to backup/"
    echo "  $0 restore             - Restore checkpoints from backup/"
    echo "========================================"
    echo ""
}

# Configuration
OUTPUT_DIR="${PROJECT_ROOT}/results/output_ensemble"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
BACKUP_DIR="$OUTPUT_DIR/backup"

case "$1" in
    status)
        echo "Checking training progress..."
        "${PYTHON_BIN}" -m model_script.train.check_progress --output_dir "$OUTPUT_DIR"
        ;;
    
    resume)
        echo "Resuming ensemble training..."
        "${SCRIPT_DIR}/run_ensemble_training.sh" --resume
        ;;
    
    clean)
        read -p "WARNING: This will delete ALL checkpoints. Are you sure? (yes/no): " confirm
        if [ "$confirm" == "yes" ]; then
            echo "Removing all checkpoints..."
            rm -rf "$CHECKPOINT_DIR"
            rm -rf "$OUTPUT_DIR/logs"
            echo ""
            echo "Checkpoints removed. Training will start from scratch."
        else
            echo "Cancelled."
        fi
        ;;
    
    clean-model)
        if [ -z "$2" ]; then
            echo "Error: Please specify model number"
            echo "Usage: $0 clean-model N"
            exit 1
        fi
        MODEL_ID=$2
        MODEL_DIR="$CHECKPOINT_DIR/model_$MODEL_ID"
        
        if [ -d "$MODEL_DIR" ]; then
            read -p "WARNING: Remove checkpoints for model $MODEL_ID? (yes/no): " confirm
            if [ "$confirm" == "yes" ]; then
                rm -rf "$MODEL_DIR"
                echo ""
                echo "Checkpoints removed for model $MODEL_ID"
            else
                echo "Cancelled."
            fi
        else
            echo "Model $MODEL_ID not found at $MODEL_DIR"
        fi
        ;;
    
    backup)
        echo "Creating backup of checkpoints..."
        mkdir -p "$BACKUP_DIR"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_FILE="$BACKUP_DIR/checkpoints_backup_$TIMESTAMP.tar.gz"
        
        if [ -d "$CHECKPOINT_DIR" ]; then
            tar -czf "$BACKUP_FILE" -C "$OUTPUT_DIR" checkpoints
            SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
            echo ""
            echo "Backup created: $BACKUP_FILE ($SIZE)"
        else
            echo "No checkpoints found to backup"
        fi
        ;;
    
    restore)
        if [ ! -d "$BACKUP_DIR" ]; then
            echo "No backup directory found"
            exit 1
        fi
        
        # Find latest backup
        LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/checkpoints_backup_*.tar.gz 2>/dev/null | head -1)
        
        if [ -z "$LATEST_BACKUP" ]; then
            echo "No backup files found"
            exit 1
        fi
        
        echo "Latest backup: $LATEST_BACKUP"
        read -p "WARNING: Restore from this backup? Current checkpoints will be overwritten. (yes/no): " confirm
        
        if [ "$confirm" == "yes" ]; then
            echo "Restoring checkpoints..."
            rm -rf "$CHECKPOINT_DIR"
            tar -xzf "$LATEST_BACKUP" -C "$OUTPUT_DIR"
            echo ""
            echo "Checkpoints restored from $LATEST_BACKUP"
        else
            echo "Cancelled."
        fi
        ;;
    
    *)
        show_usage
        ;;
esac
