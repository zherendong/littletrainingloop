"""
Language model training.
"""

import dataclasses

from training_loop import (
    TrainingConfig,
    DataProvider,
    TrainingState,
    Metrics,
    train,
)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


@dataclasses.dataclass(frozen=True)
class LanguageModelTrainingConfig:
    vocab_size: int = 1000
    dimension: int = 128
    learning_rate: float = 0.1
    seed: int = 42

    training_config: TrainingConfig = dataclasses.field(
        default_factory=lambda: TrainingConfig(
            num_epochs=50,
            eval_every_n_steps=10,
        )
    )


class DummyLanguageModel(nn.Module):
    """Simple language model: y = Wx + b"""

    def __init__(self, vocab_size: int, dimension: int, seed: int):
        super(DummyLanguageModel, self).__init__()
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, dimension)
            self.fc = nn.Linear(dimension, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens is a tensor of shape (batch_size, sequence_length) of token ids."""
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens to be 2D, got {tokens.shape}")
        if tokens.dtype not in {torch.int32, torch.int64}:
            raise ValueError(
                f"Expected tokens to be int32 or int64, got {tokens.dtype}"
            )
        x = self.embedding(tokens)
        x = self.fc(x)
        return x


class LanguageModelTrainingState(TrainingState[torch.Tensor]):
    """Training state for language model"""

    def __init__(self, model: DummyLanguageModel, config: LanguageModelTrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # linear learning rate schedule with warmup
        switch_step = config.training_config.num_epochs * 0.05
        linear_warmup = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=switch_step
        )
        linear_decay = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.training_config.num_epochs - switch_step,
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[linear_warmup, linear_decay],
            milestones=[switch_step],
        )

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def print_parameters(self) -> str:
        return f"{self.model.embedding.weight.data} {self.model.fc.weight.data} {self.model.fc.bias.data}"

    def step(self, data: torch.Tensor) -> Metrics:
        # Forward pass
        predictions = self.model(data)
        loss = self.criterion(predictions, data)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # detach loss
        loss_numpy = float(loss.detach().numpy())

        return {"loss": loss_numpy}

    def eval(self, data: torch.Tensor) -> Metrics:
        predictions = self.model(data)
        loss = self.criterion(predictions, data)
        return {"loss": float(loss.detach().numpy())}
