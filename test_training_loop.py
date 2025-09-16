#!/usr/bin/env python3
"""
Test script for the minimal training loop
"""

import torch
import pytest  # noqa: F401

import training_loop


def test_linear_model():
    """Test the LinearModel class"""
    print("Testing LinearModel...")

    from training_loop import LinearModel

    model = LinearModel(5, 2)
    x = torch.randn(3, 5)
    y = model(x)

    assert y.shape == (3, 2), f"Expected shape (3, 2), got {y.shape}"
    print("✓ LinearModel test passed")


def test_data_generation():
    """Test the data generation function"""
    print("Testing data generation...")

    from training_loop import TrainingConfig

    # Create test config
    config = TrainingConfig(num_samples=100, input_size=5, output_size=2)
    X, y, true_weights, true_bias = training_loop.generate_random_data(config)

    assert X.shape == (100, 5), f"Expected X shape (100, 5), got {X.shape}"
    assert y.shape == (100, 2), f"Expected y shape (100, 2), got {y.shape}"
    assert true_weights.shape == (5, 2), (
        f"Expected weights shape (5, 2), got {true_weights.shape}"
    )
    assert true_bias.shape == (2,), f"Expected bias shape (2,), got {true_bias.shape}"

    print("✓ Data generation test passed")


def test_training():
    """Test the training function"""
    print("Testing training function...")

    from training_loop import LinearModel, train_model, TrainingConfig

    # Create simple data
    X = torch.randn(50, 3)
    y = torch.randn(50, 1)

    # Create model
    model = LinearModel(3, 1)

    # Create test config
    config = TrainingConfig(num_epochs=5, learning_rate=0.1)

    # Train for a few epochs
    losses = train_model(model, X, y, config)

    assert len(losses) == 5, f"Expected 5 losses, got {len(losses)}"
    assert losses[-1] < losses[0], "Loss should decrease during training"

    print("✓ Training function test passed")


def test_weight_comparison():
    """Test the weight comparison function"""
    print("Testing weight comparison...")

    from training_loop import LinearModel, compare_weights

    model = LinearModel(3, 1)
    true_weights = torch.randn(3, 1)
    true_bias = torch.randn(1)

    # This should run without errors
    compare_weights(model, true_weights, true_bias)

    print("✓ Weight comparison test passed")
