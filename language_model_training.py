"""
Language model training.
"""

import argparse
import dataclasses
import os
import time
import torch.cuda.nvtx as nvtx
import null_neptune

from training_basics import (
    TrainingConfig,
    ShuffleBuffer,
    TrainingState,
    Metrics,
)
from language_model_basics import (
    DataItem,
    LanguageModelTrainingConfig,
    EvalConfig,
)
from training_loop import train
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

import stackv2_dataloader
import slimpajama_dataloader
import prng
import language_model_basics
import neptune
import transformer
import model_configs.chinchilla  # noqa: F401
import cut_cross_entropy


# Ignore_index is a magic number for cross entropy loss
# when a target token is set to this value, we ignore the
# loss for this token.
_cross_entropy_ignore_index = -100


class DummyLanguageModel(language_model_basics.LanguageModel):
    """Simple language model: y = Wx + b"""

    def __init__(self, vocab_size: int, seed: int, dimension: int = 64):
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

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_embedding_parameters(self):
        return sum(p.numel() for p in self.embedding.parameters()) + sum(
            p.numel() for p in self.fc.parameters()
        )

    def num_non_embedding_parameters(self):
        return self.num_parameters() - self.num_embedding_parameters()


class LanguageModelTrainingState(TrainingState[DataItem]):
    """Training state for language model"""

    def __init__(
        self,
        model: language_model_basics.LanguageModel,
        config: LanguageModelTrainingConfig,
    ):
        self.model = model
        self.config = config

        with prng.PRNG(config.seed + 345345):
            self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        # linear learning rate schedule with warmup
        assert config.training_config.training_steps_per_epoch is not None
        num_steps = (
            config.training_config.num_epochs
            * config.training_config.training_steps_per_epoch
        )
        switch_step = config.warmup_steps or int(num_steps * 0.05)
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
        self.training_flops_total = 0
        self.num_tokens_seen = 0

    def flops_per_step(self, batch_shape: tuple[int, int]) -> int:
        batch_size, sequence_length = batch_shape
        return (
            6 * self.model.num_non_embedding_parameters() * batch_size * sequence_length
        )

    def step(self, data: DataItem) -> Metrics:

        start = time.time()
        targets = torch.where(
            data.loss_mask == 0.0,
            _cross_entropy_ignore_index,
            data.targets,
        )

        with nvtx.range("train_step", color="blue"):
            loss = self.model.compute_loss(data.inputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()
            # detach loss
            loss_numpy = float(loss.to(torch.float32).detach().cpu().numpy())

        step_time = time.time() - start
        flops_this_step = self.flops_per_step(data.inputs.shape)  # type: ignore
        self.training_flops_total += flops_this_step
        self.num_tokens_seen += data.inputs.numel()

        return {
            "loss": loss_numpy,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "pflops_total": self.training_flops_total / 1e15,
            "tflops_per_second": flops_this_step / step_time / 1e12,
            "step_time_seconds": step_time,
            "num_tokens": self.num_tokens_seen,
        }

    def eval(self, data: DataItem) -> Metrics:
        with torch.no_grad():
            targets = torch.where(
                data.loss_mask == 0.0,
                _cross_entropy_ignore_index,
                data.targets,
            )
            loss = self.model.compute_loss(data.inputs, targets)
            loss = float(loss.to(torch.float32).detach().cpu().numpy())
        return {"loss": loss}

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def num_non_embedding_parameters(self) -> int:
        return self.model.num_non_embedding_parameters()

    def get_training_pflops(self) -> float:
        return self.training_flops_total / 1e15

    def get_training_tokens_seen(self) -> int:
        return self.num_tokens_seen


def train_language_model(
    config: LanguageModelTrainingConfig,
    *,
    neptune_run,
    dataset: str = "slimpajama",
):
    """Train a language model using configuration object"""
    # Create model
    with prng.PRNG(config.seed + 123123):
        model = transformer.TransformerModel(config.vocab_size, config.model_config)

    if config.training_config.training_steps_per_epoch is None:
        # Chinchilla-optimal amount of data, which is 20 tokens per parameter
        num_parameters = model.num_non_embedding_parameters()
        num_tokens_per_step = config.batch_size * config.sequence_length
        chinchilla_optimal_steps = int(20 * num_parameters / num_tokens_per_step)
        print(f"Using Chinchilla-optimal number of steps: {chinchilla_optimal_steps}")
        config = dataclasses.replace(
            config,
            training_config=dataclasses.replace(
                config.training_config,
                training_steps_per_epoch=chinchilla_optimal_steps,
            ),
        )

    # Create training state
    with prng.PRNG(config.seed + 234234):
        state = LanguageModelTrainingState(model, config)
    # Create data generator
    if dataset == "stackv2":
        train_dataset = stackv2_dataloader.create_stackv2_dataloader(config)
        eval_datasets = [
            stackv2_dataloader.create_stackv2_dataloader(config, split="validation")
        ]
    elif dataset == "slimpajama":
        train_dataset = slimpajama_dataloader.create_slimpajama_dataloader(config)
        eval_datasets = [
            slimpajama_dataloader.create_slimpajama_dataloader(
                config, split="validation"
            )
        ]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if config.shuffle_buffer_size:
        train_dataset = ShuffleBuffer(
            config.training_config,
            config.shuffle_buffer_size,
            train_dataset,
            name=train_dataset.get_name(),
        )

    neptune_run["num_parameters"] = state.num_parameters()
    neptune_run["num_non_embedding_parameters"] = state.num_non_embedding_parameters()
    neptune_run["config"] = dataclasses.asdict(config)
    # Train the model
    losses = train(
        state,
        train_dataset,
        config=config.training_config,
        eval_config=config.eval_config,
        eval_data_providers=eval_datasets,
        neptune_run=neptune_run,
    )
    return losses


def run(
    model_config_str: str,
    description: str,
    use_neptune: bool = False,
    profile_only: bool = False,
):
    # Add device detection at the top of your training function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_device(device)

    model_config = transformer.transformer_config_registry.get(model_config_str)
    config = LanguageModelTrainingConfig(
        vocab_size=100277,
        warmup_steps=100,
        learning_rate=0.0005,
        batch_size=256,
        sequence_length=512,
        shuffle_buffer_size=100,
        training_config=TrainingConfig(
            num_epochs=1,
            training_steps_per_epoch=(
                None if not profile_only else 10
            ),  # None defaults to Chinchilla
            seed=42,
        ),
        eval_config=EvalConfig(
            every_n_steps=100,
            steps=5,
            batch_size=256,
            sequence_length=512,
        ),
        model_config=model_config,
    )

    if use_neptune:
        print("Using neptune")
        load_dotenv(dotenv_path=os.path.expanduser("~/.neptune/.env"))
        neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
        neptune_run = neptune.init_run(
            project="markusrabeworkspace/training-exploration",
            api_token=neptune_api_token,
            description=description,
        )
        neptune_run["model_config"] = model_config_str
    else:
        neptune_run = null_neptune.NullNeptuneRun()
    try:
        losses = train_language_model(config, neptune_run=neptune_run)
        print(f"Losses: {losses}")
    finally:
        neptune_run.stop()


if __name__ == "__main__":

    # command line args, including name
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_neptune", action="store_true", default=False)
    parser.add_argument("--model_config", type=str, default="chinchilla-44m")
    parser.add_argument("--profile_only", action="store_true", default=False)
    parser.add_argument("--description", "-d", type=str, default=None)
    args = parser.parse_args()

    assert (
        args.description is not None or args.profile_only
    ), "Must provide a description"

    run(
        use_neptune=not args.no_neptune and not args.profile_only,
        model_config_str=args.model_config,
        profile_only=args.profile_only,
        description=args.description,
    )
