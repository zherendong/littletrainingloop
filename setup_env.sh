#!/bin/bash

# WARNING. THIS IS A BRUTAL HACK.

# Setup script for littletrainingloop environment
# Run this on each machine to fix package conflicts

set -e  # Exit on error

echo "=== Removing system TensorFlow ==="
sudo apt remove -y python3-tensorflow-cuda || echo "TensorFlow not installed via apt, skipping"

echo "=== Installing PyTorch 2.7.0 with CUDA 12.8 (to match system flash-attn) ==="
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== Force reinstalling packages with numpy compatibility issues ==="
pip install --force-reinstall pandas scipy

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
import numpy as np
print(f'NumPy version: {np.__version__}')
"

echo "=== Setup complete ==="
