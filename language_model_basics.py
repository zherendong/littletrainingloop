"""
Basic classes for language model training.
"""

import abc
import dataclasses
from typing import Any

import torch
import training_basics
import numpy as np


@dataclasses.dataclass
class LMData:
    # token ids of shape (batch_size, sequence_length)
    inputs: np.ndarray
    # token ids of shape (batch_size, sequence_length)
    targets: np.ndarray
    # 0/1 mask of shape (batch_size, sequence_length)
    loss_mask: np.ndarray
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EvalConfig(training_basics.EvalConfig):
    batch_size: int = 256
    sequence_length: int = 512


@dataclasses.dataclass(frozen=True)
class LanguageModelTrainingConfig:
    name: str = "default_name"
    vocab_size: int = 100277
    warmup_steps: int = 500  # 0 means 5% of training steps
    learning_rate: float | None = None  # None means auto-select based on model size
    seed: int = 42
    batch_size: int = 16
    sequence_length: int = 512
    shuffle_buffer_size: int = 100
    model_config: Any = None
    separate_data_with_eot: bool = True
    dataset: str = "slimpajama"

    # Growing MLP schedule
    add_block_at_steps: list[int] | None = None
    block_schedule_steps: list[int] | None = None

    adam_eps: float = 1e-7
    adam_betas: tuple[float, float] = (0.9, 0.995)
    weight_decay: float = 0.1

    training_config: training_basics.TrainingConfig = dataclasses.field(
        default_factory=lambda: training_basics.TrainingConfig()
    )
    eval_config: EvalConfig = dataclasses.field(default_factory=lambda: EvalConfig())
    chinchilla_factor: float = 1.0

    # Optional path to write MLP activation stats CSV (None = disabled).
    # Stats are collected every 500 steps via mlp_tracking.MLPTracker.
    mlp_tracking_output: str | None = None


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
