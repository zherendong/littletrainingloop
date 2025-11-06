#!/bin/bash
# Run your specific experiment:
# - Model size: 117m
# - Weight decay: 0.1
# - use_depth_scaling: False
# - use_proper_init: True
# - chinchilla_factor: 1.0
# - learning_rate: 0.003
# - warmup_steps: 500

set -e  # Exit on error

python run_custom_experiment.py \
    --model_size 117m \
    --neptune_tags silu-init-zheren \
    --learning_rate 0.003 \
    --weight_decay 0.1 \
    --warmup_steps 500 \
    --chinchilla_factor 1.0 \
    --use_proper_init \
    --no_depth_scaling \
    "$@"

echo ""
echo "Experiment complete!"

