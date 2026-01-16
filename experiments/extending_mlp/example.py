"""
Example: Training with a GrowingMLP that adds blocks at specific steps.

This demonstrates:
1. Creating a GrowingMLP with initial blocks
2. Training loop that adds blocks at specific steps
3. Per-block learning rate warmup

Example call:
python example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from growing_mlp import GrowingMLP
from block_scheduler import create_block_scheduler

torch.set_default_device("cuda")


def example_growing_mlp_training(static: bool = False):
    """Simple example of training with block addition."""

    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    input_size = 128
    block_size = 512  # small for testing
    num_blocks = 2
    block_training_steps = 1000
    total_steps = int(block_training_steps * num_blocks)
    add_block_at_steps = [block_training_steps * i for i in range(1, num_blocks)]
    if static:
        block_size += len(add_block_at_steps) * block_size
        add_block_at_steps = []
        total_steps = int(0.75 * total_steps)

    print(f"Device: {device}, dtype: {dtype}")

    # Create model
    model = GrowingMLP(
        dtype=dtype,
        input_size=input_size,
        block_size=block_size,
        initial_blocks=1,
        glu=False,
        pairwise_cancelling_init=True,
    ).to(device)
    model.init_weights()

    print(
        f"Initial model: {model.num_blocks} block(s), {model.total_inner_size} KV pairs"
    )

    # Create optimizer with initial parameters
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    # Create scheduler with per-block warmup
    # Each block follows: warmup (100 steps) -> decay (remaining steps)
    scheduler = create_block_scheduler(
        optimizer,
        total_steps=int(block_training_steps * 1.0) if not static else total_steps,
        warmup_steps=100,
    )

    # Simple target: identity function (for testing)
    criterion = nn.MSELoss()

    # Training loop
    for step in range(total_steps):
        # Check if we should add a block
        if step in add_block_at_steps:
            print(f"\n=== Step {step}: Adding block ===")
            new_params = model.add_block()
            # Move new block to device
            model.blocks[-1].to(device)
            # Add to scheduler (which adds to optimizer) - starts at step 0 of its own schedule
            scheduler.add_param_group(new_params)
            print(
                f"Now have {model.num_blocks} blocks, {model.total_inner_size} KV pairs"
            )
            print(f"Param groups in optimizer: {len(optimizer.param_groups)}")

        # Generate random data
        x = torch.randn(8, 16, input_size, device=device, dtype=dtype)
        target = x  # identity target for simplicity

        # Forward
        output = model(x)
        loss = criterion(output.float(), target.float())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if step % 100 == 0:
            lrs = scheduler.get_last_lr()
            lr_str = ", ".join(f"{lr:.6f}" for lr in lrs)
            print(f"Step {step}: loss={loss.item():.6f}, LRs=[{lr_str}]")

    print(f"\nFinal: {model.num_blocks} blocks, {model.total_inner_size} KV pairs")


def test_forward_pass():
    """Test that forward pass works with multiple blocks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = GrowingMLP(
        dtype=dtype,
        input_size=64,
        block_size=32,
        initial_blocks=2,
    ).to(device)
    model.init_weights()

    x = torch.randn(4, 8, 64, device=device, dtype=dtype)
    out = model(x)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print("Forward pass test: PASSED")


def test_add_block():
    """Test adding blocks dynamically."""
    model = GrowingMLP(
        dtype=torch.float32,
        input_size=64,
        block_size=32,
        initial_blocks=1,
    )
    model.init_weights()

    assert model.num_blocks == 1
    assert model.total_inner_size == 32

    # Add a block
    new_params = list(model.add_block())
    assert model.num_blocks == 2
    assert model.total_inner_size == 64
    assert len(new_params) > 0, "Should return new parameters"

    # Forward still works
    x = torch.randn(2, 4, 64)
    out = model(x)
    assert out.shape == x.shape

    print("Add block test: PASSED")


if __name__ == "__main__":
    # print("=" * 50)
    # print("Test: Forward pass")
    # print("=" * 50)
    # test_forward_pass()

    # print("\n" + "=" * 50)
    # print("Test: Add block")
    # print("=" * 50)
    # test_add_block()

    print("\n" + "=" * 50)
    print("Example: Training with static MLP")
    print("=" * 50)
    example_growing_mlp_training(static=True)

    print("\n" + "=" * 50)
    print("Example: Training with block addition")
    print("=" * 50)
    example_growing_mlp_training()
