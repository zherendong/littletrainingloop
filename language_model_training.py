"""
Language model training.
"""

import argparse
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
import neptune
from dotenv import load_dotenv
from torch.utils import flop_counter

import stackv2_dataloader
import slimpajama_dataloader
from transformer import TransformerModel, TransformerConfig
import prng


class DummyLanguageModel(nn.Module):
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


class LanguageModelTrainingState(TrainingState[DataItem]):
    """Training state for language model"""

    def __init__(self, model: nn.Module, config: LanguageModelTrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        with prng.PRNG(config.seed + 345345):
            self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # linear learning rate schedule with warmup
        num_steps = (
            config.training_config.num_epochs
            * config.training_config.training_steps_per_epoch
        )
        switch_step = config.warmup_steps or num_steps * 0.05
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
        self.fcounter = flop_counter.FlopCounterMode(depth=4, display=False)
        self.training_flops_total = 0
        self.num_tokens_total = 0

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def num_non_embedding_parameters(self):
        return sum(p.numel() for p in self.model.transformer_blocks.parameters())

    def step(self, data: DataItem) -> Metrics:

        start = time.time()
        with self.fcounter:
            # Forward pass
            with nvtx.range("forward", color="blue"):
                predictions = self.model(data.inputs)

            with nvtx.range("loss", color="green"):
                # apply loss mask
                predictions = predictions * data.loss_mask.unsqueeze(-1)

                # flatten batch and sequence length for cross entropy
                predictions = predictions.view(-1, self.config.vocab_size)
                targets = data.targets.view(-1).long()
                loss = self.criterion(predictions, targets)

            with nvtx.range("backward", color="red"):
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

        # detach loss
        loss_numpy = float(loss.detach().cpu().numpy())

        step_time = time.time() - start
        flops_this_step = self.fcounter.get_total_flops()
        self.training_flops_total += flops_this_step
        num_tokens_this_step = data.inputs.numel()
        self.num_tokens_total += num_tokens_this_step

        return {
            "loss": loss_numpy,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "pflops_total": self.training_flops_total / 1e15,
            "tflops_per_second": flops_this_step / step_time / 1e12,
            "step_time_seconds": step_time,
            "num_tokens": self.num_tokens_total,
        }

    def eval(self, data: DataItem) -> Metrics:
        predictions = self.model(data.inputs)

        # apply loss mask
        predictions = predictions * data.loss_mask.unsqueeze(-1)

        # flatten batch and sequence length for cross entropy
        predictions = predictions.view(-1, self.config.vocab_size)
        targets = data.targets.view(-1).long()
        loss = self.criterion(predictions, targets)
        loss = float(loss.detach().cpu().numpy())
        return {
            "loss": loss,
            "loss_vs_tokens": (loss, self.num_tokens_total),
            "loss_vs_pflops_total": (loss, self.training_flops_total / 1e15),
        }


def train_language_model(
    config: LanguageModelTrainingConfig,
    dataset: str = "slimpajama",
    neptune_run=None,
):
    """Train a language model using configuration object"""
    # Create model
    with prng.PRNG(config.seed + 123123):
        model = TransformerModel(config.vocab_size, config.model_config)
    # Create training state
    with prng.PRNG(config.seed + 234234):
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

    if config.shuffle_buffer_size:
        train_dataset = ShuffleBuffer(
            config.training_config,
            config.shuffle_buffer_size,
            train_dataset,
            name=train_dataset.get_name(),
        )

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


def run(use_neptune: bool):
    # Add device detection at the top of your training function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_default_device(device)

    if use_neptune:
        load_dotenv(dotenv_path=os.path.expanduser("~/.neptune/.env"))
        neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]
        neptune_run = neptune.init_run(
            project="markusrabeworkspace/training-exploration",
            api_token=neptune_api_token,
        )
    else:
        neptune_run = null_neptune.NullNeptuneRun()
    try:
        config = LanguageModelTrainingConfig(
            vocab_size=100277,
            warmup_steps=100,
            learning_rate=0.001,
            batch_size=64,
            sequence_length=512,
            shuffle_buffer_size=100,
            training_config=TrainingConfig(
                num_epochs=1,
                training_steps_per_epoch=50000,
                seed=42,
            ),
            eval_config=EvalConfig(
                every_n_steps=50,
                steps=10,
                batch_size=64,
                sequence_length=512,
            ),
            model_config=TransformerConfig(
                num_layers=6,
                num_heads=4,
                num_heads_kv=4,
                head_dim=64,
                mlp_inner_size=512,
                embedding_size=128,
            ),
        )
        losses = train_language_model(config, neptune_run=neptune_run)
        print(f"Losses: {losses}")
    finally:
        neptune_run.stop()


if __name__ == "__main__":

    # command line args, including name
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_neptune", type=bool, default=False)
    args = parser.parse_args()

    run(use_neptune=args.use_neptune)
