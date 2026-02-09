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


def example_mnist_training(static: bool = False, use_pairwise_init=True):
    """MNIST training experiment to test growing MLP on real data."""
    from torchvision import datasets, transforms
    
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    input_size = 784  # 28x28 flattened
    output_size = 10  # digits 0-9
    block_size = 512
    num_blocks = 4
    block_training_steps = 400
    total_steps = block_training_steps * num_blocks
    add_block_at_steps = [block_training_steps * i for i in range(1, num_blocks)]
    
    if static:
        block_size = block_size * num_blocks
        add_block_at_steps = []
        average_blocks_per_step = sum(range(1, num_blocks+1)) / num_blocks
        total_steps = int(average_blocks_per_step * total_steps / num_blocks)

    print(f"Static: {static}, block_size: {block_size}")
    
    # Load MNIST
    train_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    # Create model
    model = GrowingMLP(
        dtype=dtype,
        input_size=input_size,
        output_size=output_size,
        block_size=block_size,
        initial_blocks=1,
        glu=False,
        pairwise_cancelling_init=use_pairwise_init,
        copy_most_active_init=True,
    ).to(device)
    model.init_weights()
    
    print(f"Initial model: {model.num_blocks} block(s), {model.total_inner_size} hidden units")
    print(f"pairwise cancelling {'True' if use_pairwise_init else 'False'}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = create_block_scheduler(
        optimizer,
        total_steps=block_training_steps if not static else total_steps,
        warmup_steps=10,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    step = 0
    train_iter = iter(train_loader)
    
    while step < total_steps:
        # Check if we should add a block
        if step in add_block_at_steps:
            # evaluation on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    x = images.view(images.size(0), -1).to(device=device, dtype=dtype)
                    labels = labels.to(device)
                    logits = model(x)
                    correct += (logits.argmax(dim=-1) == labels).sum().item()
                    total += labels.size(0)
            
            print(f"\nTest accuracy: {correct/total:.4f}")
            print(f"{model.num_blocks} blocks, {model.total_inner_size} hidden units")
            
            model.train()
            print(f"\n=== Step {step}: Adding block ===")
            new_params = model.add_block()
            model.blocks[-1].to(device)
            scheduler.add_param_group(new_params)
            print(f"Now have {model.num_blocks} blocks, {model.total_inner_size} hidden units")
        
        # Get batch (cycle through data)
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
        
        x = images.view(images.size(0), -1).to(device=device, dtype=dtype)
        labels = labels.to(device)
        
        # Forward
        logits = model(x)
        loss = criterion(logits.float(), labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        if step % 10 == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            lrs = scheduler.get_last_lr()
            lr_str = ", ".join(f"{lr:.6f}" for lr in lrs)
            print(f"Step {step}: loss={loss.item():.4f}, acc={acc:.3f}, LRs=[{lr_str}]")
        
        step += 1
    
    # Final evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.view(images.size(0), -1).to(device=device, dtype=dtype)
            labels = labels.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    
    print(f"\nFinal test accuracy: {correct/total:.4f}")
    print(f"Final: {model.num_blocks} blocks, {model.total_inner_size} hidden units")

def example_growing_mlp_training(static: bool = False, use_pairwise_init: bool = True):
    """Simple example of training with block addition."""
    # torch.set_default_device("cuda")
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
        pairwise_cancelling_init=use_pairwise_init,
    ).to(device)
    model.init_weights()

    print(
        f"Initial model: {model.num_blocks} block(s), {model.total_inner_size} KV pairs"
    )
    print(f"pairwise cancelling {'True' if use_pairwise_init else 'False'}")

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

    # # 1. Experiment with random data
    # print("\n" + "=" * 50)
    # print("Example: Training with static MLP")
    # print("=" * 50)
    # example_growing_mlp_training(static=True, use_pairwise_init=True)

    # print("\n" + "=" * 50)
    # print("Example: Training with block addition")
    # print("=" * 50)
    # example_growing_mlp_training(use_pairwise_init=True)
    
    # 2. Experiment with mnist
    print("\n" + "=" * 50)
    print("MNIST: Static training, paired init")
    print("=" * 50)
    example_mnist_training(static=True, use_pairwise_init=True)
   
    # print("\n" + "=" * 50)
    # print("MNIST: Static training, no paired weights")
    # print("=" * 50)
    # example_mnist_training(static=True, use_pairwise_init=False)
   
    print("\n" + "=" * 50)
    print("MNIST: Growing training")
    print("=" * 50)
    example_mnist_training(static=False)
