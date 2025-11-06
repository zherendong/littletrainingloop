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

# python run_custom_experiment.py \
#     --model_size 117m \
#     --neptune_tags silu-init-zheren \
#     --learning_rate 0.003 \
#     --weight_decay 0.1 \
#     --warmup_steps 500 \
#     --chinchilla_factor 1.0 \
#     --use_proper_init \
#     --no_depth_scaling \
#     "$@"

# best: lr0.003 wu500 wd0.1 init no_depth

echo ""

python run_custom_experiment.py --model_size 117m --neptune_tags silu-init-zheren --learning_rate 0.003 --weight_decay 0.1 --warmup_steps 100 --chinchilla_factor 1.0 --no_depth_scaling --description "Training chinchilla-117m with lr=0.003, wd=0.1, warmup=100, proper_init=True, depth_scaling=False. This is to compare with the previous run TRAIN-830."

echo ""

python run_custom_experiment.py --model_size 117m --neptune_tags silu-init-zheren --learning_rate 0.003 --weight_decay 0.1 --warmup_steps 100 --chinchilla_factor 1.0 --use_depth_scaling --description "Training chinchilla-117m with lr=0.003, wd=0.1, warmup=100, proper_init=True, depth_scaling=True. This is to compare with the previous run TRAIN-830."

echo ""

python run_custom_experiment.py --model_size 117m --neptune_tags silu-init-zheren --learning_rate 0.003 --weight_decay 0.1 --warmup_steps 500 --chinchilla_factor 1.0 --use_depth_scaling --description "Training chinchilla-117m with lr=0.003, wd=0.1, warmup=500, proper_init=True, depth_scaling=True. This is to compare with the previous run TRAIN-830."


echo ""

python run_custom_experiment.py --model_size 117m --neptune_tags silu-init-zheren --learning_rate 0.002 --weight_decay 0.1 --warmup_steps 500 --chinchilla_factor 1.0 --no_depth_scaling --description "Training chinchilla-117m with lr=0.002, wd=0.1, warmup=500, proper_init=True, depth_scaling=True. This is to compare with the previous run TRAIN-830."

echo ""

echo "Experiment complete!"

