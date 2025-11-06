#!/bin/bash
# Quick script to run depth scaling experiment on chinchilla-117m
# By default, trains the model twice:
# 1. Without depth scaling (baseline)
# 2. With depth scaling
# Use --depth_scaling_only to train only with depth scaling

set -e  # Exit on error

# Default values
DRY_RUN=""
NEPTUNE_TAGS="depth_scaling_experiment"
CHINCHILLA_FACTOR=""
DEPTH_SCALING_MODE="--depth_scaling_variants"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --tags)
            NEPTUNE_TAGS="$2"
            shift 2
            ;;
        --chinchilla_factor)
            CHINCHILLA_FACTOR="--chinchilla_factor $2"
            shift 2
            ;;
        --depth_scaling_only)
            DEPTH_SCALING_MODE="--depth_scaling_only"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry_run                  Run in dry-run mode (preview only)"
            echo "  --tags TAG                 Set Neptune tags (default: depth_scaling_experiment)"
            echo "  --chinchilla_factor FLOAT  Multiplier for Chinchilla-optimal training (default: 1.0)"
            echo "                             Use 2.0 for 40 tokens/param, 0.5 for 10 tokens/param"
            echo "  --depth_scaling_only       Train with depth scaling only (skip baseline)"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Dry run to preview (trains both variants)"
            echo "  $0 --dry_run"
            echo ""
            echo "  # Run the actual experiment (trains both variants)"
            echo "  $0"
            echo ""
            echo "  # Train with depth scaling only (skip baseline)"
            echo "  $0 --depth_scaling_only"
            echo ""
            echo "  # Run with custom tags"
            echo "  $0 --tags my_experiment v1"
            echo ""
            echo "  # Run with 2x Chinchilla-optimal data (40 tokens per parameter)"
            echo "  $0 --chinchilla_factor 2.0"
            echo ""
            echo "  # Quick test with 0.5x Chinchilla-optimal data, depth scaling only"
            echo "  $0 --chinchilla_factor 0.5 --depth_scaling_only --dry_run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running depth scaling experiment on chinchilla-117m"
echo "Neptune tags: $NEPTUNE_TAGS"
if [ "$DEPTH_SCALING_MODE" = "--depth_scaling_only" ]; then
    echo "Mode: Depth scaling only (skip baseline)"
else
    echo "Mode: Compare both variants (with and without depth scaling)"
fi
if [ -n "$CHINCHILLA_FACTOR" ]; then
    echo "Chinchilla factor: ${CHINCHILLA_FACTOR#--chinchilla_factor }"
else
    echo "Chinchilla factor: 1.0 (default)"
fi
if [ -n "$DRY_RUN" ]; then
    echo "Dry run: Yes (preview only)"
else
    echo "Dry run: No (actual training)"
fi
echo ""

python train_chinchilla_series.py \
    --neptune_tags $NEPTUNE_TAGS \
    --model_sizes 117m \
    $DEPTH_SCALING_MODE \
    $CHINCHILLA_FACTOR \
    $DRY_RUN

echo ""
echo "Experiment complete!"

