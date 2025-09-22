"""
Language model training.
"""

import os

from training_basics import (
    TrainingConfig,
    TrainingState,
    Metrics,
)
from language_model_basics import (
    DataItem,
    LanguageModelTrainingConfig,
)
from training_loop import train
import torch
import torch.nn as nn
import torch.optim as optim
import neptune
from dotenv import load_dotenv

import stackv2_dataloader
import slimpajama_dataloader


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

        # apply loss mask
        predictions = predictions * data.loss_mask.unsqueeze(-1)

        # flatten batch and sequence length for cross entropy
        predictions = predictions.view(-1, self.model.vocab_size)
        targets = data.targets.view(-1).long()
        loss = self.criterion(predictions, targets)
        return {"loss": float(loss.detach().numpy())}


def train_language_model(
    config: LanguageModelTrainingConfig,
    dataset: str = "slimpajama",
    neptune_run=None,
):
    """Train a language model using configuration object"""
    # Create model
    model = DummyLanguageModel(config.vocab_size, config.dimension, config.seed)
    # Create training state
    state = LanguageModelTrainingState(model, config)
    # Create data generator
    if dataset == "stackv2":
        train_dataset = stackv2_dataloader.create_stackv2_dataloader(config)
        eval_datasets = []
    elif dataset == "slimpajama":
        train_dataset = slimpajama_dataloader.create_slimpajama_dataloader(config)
        eval_datasets = [
            slimpajama_dataloader.create_slimpajama_dataloader(
                config, split="validation"
            )
        ]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # Train the model
    losses = train(
        state,
        train_dataset,
        config.training_config,
        eval_data_providers=eval_datasets,
        neptune_run=neptune_run,
    )
    return losses


def run():
    load_dotenv(dotenv_path=os.path.expanduser("~/.neptune/.env"))
    neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
    neptune_run = neptune.init_run(
        project="markusrabeworkspace/training-exploration",
        # name="language-model-training",
        api_token=neptune_api_token,
    )
    try:
        config = LanguageModelTrainingConfig(
            vocab_size=100277,
            dimension=64,
            learning_rate=0.01,
            seed=42,
            batch_size=32,
            sequence_length=64,
            training_config=TrainingConfig(
                num_epochs=1,
                training_steps_per_epoch=500,
                eval_every_n_steps=50,
                eval_steps=10,
            ),
        )
        losses = train_language_model(config, neptune_run=neptune_run)
        print(f"Losses: {losses}")
    finally:
        neptune_run.stop()


if __name__ == "__main__":
    run()
