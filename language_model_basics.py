"""
Basic classes for language model training.
"""

import dataclasses
from typing import Any

import torch
import training_basics


@dataclasses.dataclass
class DataItem:
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EvalConfig(training_basics.EvalConfig):
    batch_size: int
    sequence_length: int


@dataclasses.dataclass(frozen=True)
class LanguageModelTrainingConfig:
    vocab_size: int = 100277
    dimension: int = 64
    warmup_steps: int = 0  # 0 means 5% of training steps
    learning_rate: float = 0.1
    seed: int = 42
    batch_size: int = 16
    sequence_length: int = 256
    shuffle_buffer_size: int = 0
    model_config: Any = None

    training_config: training_basics.TrainingConfig = dataclasses.field(
        default_factory=lambda: training_basics.TrainingConfig()
    )
    eval_config: EvalConfig = dataclasses.field(default_factory=lambda: EvalConfig())
