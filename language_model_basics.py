"""
Basic classes for language model training.
"""

import abc
import dataclasses
from typing import Any

import torch
import training_basics
import numpy as np

# Ignore_index is a magic number for cross entropy loss
# when a target token is set to this value, we ignore the
# loss for this token.
cross_entropy_ignore_index = -100


@dataclasses.dataclass
class LMData:
    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EvalConfig(training_basics.EvalConfig):
    batch_size: int
    sequence_length: int


@dataclasses.dataclass(frozen=True)
class LanguageModelTrainingConfig:
    vocab_size: int = 100277
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
    eval_config: EvalConfig = dataclasses.field(
        default_factory=lambda: EvalConfig(
            every_n_steps=100,
            steps=5,
            batch_size=128,
            sequence_length=256,
        )
    )


class LanguageModel(abc.ABC, torch.nn.Module):

    @abc.abstractmethod
    def num_parameters(self) -> int:
        pass

    @abc.abstractmethod
    def num_embedding_parameters(self) -> int:
        pass

    @abc.abstractmethod
    def num_non_embedding_parameters(self) -> int:
        pass

    @abc.abstractmethod
    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass
