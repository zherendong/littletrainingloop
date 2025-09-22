"""
Language model training.
"""

import dataclasses
from typing import Any

from training_loop import (
    TrainingConfig,
    TrainingState,
    Metrics,
    train,
)
import torch
import torch.nn as nn
import torch.optim as optim

import language_model_dataloader
import stackv2_dataloader


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
        default_factory=lambda: TrainingConfig(
            num_epochs=1,
            training_steps_per_epoch=100,
            eval_every_n_steps=10,
        )
    )


class DummyLanguageModel(nn.Module):
    """Simple language model: y = Wx + b"""

    def __init__(self, vocab_size: int, dimension: int, seed: int):
        super(DummyLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.dimension = dimension
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

        batch_size = tokens.shape[0]
        sequence_length = tokens.shape[1]
        assert x.shape == (batch_size, sequence_length, self.dimension)

        x = self.fc(x)

        assert x.shape == (batch_size, sequence_length, self.vocab_size)
        return x


class LanguageModelTrainingState(TrainingState[DataItem]):
    """Training state for language model"""

    def __init__(self, model: DummyLanguageModel, config: LanguageModelTrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # linear learning rate schedule with warmup
        num_steps = (
            config.training_config.num_epochs
            * config.training_config.training_steps_per_epoch
        )
        switch_step = num_steps * 0.05
        linear_warmup = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=switch_step
        )
        linear_decay = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_steps - switch_step,
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

    def step(self, data: DataItem) -> Metrics:
        # Forward pass
        predictions = self.model(data.inputs)

        # apply loss mask
        predictions = predictions * data.loss_mask.unsqueeze(-1)

        # flatten batch and sequence length for cross entropy
        predictions = predictions.view(-1, self.model.vocab_size)
        targets = data.targets.view(-1).long()
        loss = self.criterion(predictions, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # detach loss
        loss_numpy = float(loss.detach().numpy())

        return {
            "loss": loss_numpy,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def eval(self, data: DataItem) -> Metrics:
        predictions = self.model(data.inputs)
        loss = self.criterion(predictions, data.targets)
        return {"loss": float(loss.detach().numpy())}


def train_language_model(config: LanguageModelTrainingConfig):
    """Train a language model using configuration object"""
    # Create model
    model = DummyLanguageModel(config.vocab_size, config.dimension, config.seed)
    # Create training state
    state = LanguageModelTrainingState(model, config)
    # Create data generator
    raw_data_loader = language_model_dataloader.JSONLDataLoader(
        config, "data/stackv2_long"
    )
    tokenizer = language_model_dataloader._construct_default_tokenizer()
    if tokenizer.n_vocab > config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.n_vocab}) is larger than the model vocab size ({config.vocab_size})"
        )
    tokenized_data_loader = language_model_dataloader.TokenizedDataLoader(
        config, raw_data_loader, tokenizer, stackv2_dataloader.extract
    )
    batched_data_loader = language_model_dataloader.BatchedDataLoader(
        config, tokenized_data_loader, tokenizer
    )
    # Train the model
    losses = train(
        state,
        batched_data_loader,
        config.training_config,
    )
    return losses


def run():
    config = LanguageModelTrainingConfig(
        vocab_size=100277,
        dimension=64,
        learning_rate=0.1,
        seed=42,
        batch_size=16,
        sequence_length=256,
        training_config=TrainingConfig(
            num_epochs=1,
            training_steps_per_epoch=100,
            eval_every_n_steps=10,
        ),
    )
    losses = train_language_model(config)
    print(f"Losses: {losses}")


if __name__ == "__main__":
    run()
