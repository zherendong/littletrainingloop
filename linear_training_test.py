"""
Test script for the training loop using the linear model.
"""

import pytest  # noqa: F401
import torch
from linear_training import (
    DataItem,
    LinearModel,
    LinearModelTrainingState,
    LinearTrainingConfig,
    RandomLinearDataGenerator,
    train_linear_model,
)
from training_basics import TrainingConfig, EvalConfig


def test_linear_model():
    """Test the LinearModel class"""

    model = LinearModel(5, 2)
    x = torch.randn(3, 5)
    y = model(x)

    assert y.shape == (3, 2), f"Expected shape (3, 2), got {y.shape}"


def test_data_generation():
    """Test the data generation function"""

    config = LinearTrainingConfig(num_samples=100, input_size=5, output_size=2)
    data_generator = RandomLinearDataGenerator(config)
    data = next(data_generator.generate())

    assert isinstance(data, DataItem), f"Expected DataItem, got {type(data)}"
    assert data.inputs.shape == (
        100,
        5,
    ), f"Expected X shape (100, 5), got {data.inputs.shape}"
    assert data.targets.shape == (
        100,
        2,
    ), f"Expected y shape (100, 2), got {data.targets.shape}"
    assert data.metadata["true_weights"].shape == (
        5,
        2,
    ), f"Expected weights shape (5, 2), got {data.metadata['true_weights'].shape}"
    assert data.metadata["true_bias"].shape == (
        2,
    ), f"Expected bias shape (2,), got {data.metadata['true_bias'].shape}"


def test_training():
    """Test the training function"""

    # Set seed for reproducible test
    torch.manual_seed(42)

    config = LinearTrainingConfig(
        input_size=50,
        output_size=10,
        num_samples=10,
        learning_rate=0.3,
        training_config=TrainingConfig(num_epochs=10),
        eval_config=EvalConfig(every_n_steps=100),
    )
    train_linear_model(config)
