"""
Basic classes for language model training.
"""

import dataclasses
from typing import Any

import torch
from training_basics import TrainingConfig


@dataclasses.dataclass
class DataItem:
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class LanguageModelTrainingConfig:
    vocab_size: int = 100277
    dimension: int = 64
    learning_rate: float = 0.1
    seed: int = 42
    batch_size: int = 16
    sequence_length: int = 256

    training_config: TrainingConfig = dataclasses.field(
        default_factory=lambda: TrainingConfig()
    )
