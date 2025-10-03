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
from torch.utils import flop_counter

import stackv2_dataloader
import slimpajama_dataloader
from transformer import TransformerModel, TransformerConfig
import prng
import language_model_basics
import neptune


# Ignore_index is a magic number for cross entropy loss
# when a target token is set to this value, we ignore the
# loss for this token.
_cross_entropy_ignore_index = -100


class DummyLanguageModel(nn.Module, language_model_basics.LanguageModel):
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=_cross_entropy_ignore_index)

        with prng.PRNG(config.seed + 345345):
            self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        # linear learning rate schedule with warmup
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
        self.fcounter = flop_counter.FlopCounterMode(depth=4, display=False)
        self.training_flops_total = 0
        self.non_emb_training_flops_total = 0
        self.num_tokens_seen = 0

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

                # apply loss mask; more efficient?
                # targets = torch.where(
                #     data.loss_mask.view(-1) == 0.0,
                #     _cross_entropy_ignore_index,
                #     targets,
                # )
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
        # estimated
        emb_flops = (
            self.config.batch_size
            * self.config.sequence_length
            * self.config.vocab_size
            * self.config.model_config.embedding_size
            * 2  # multiply-add
            * 3  # backward uses twice the flops of forward
        )
        self.non_emb_training_flops_total += flops_this_step - emb_flops
        self.num_tokens_seen += data.inputs.numel()

        return {
            "loss": loss_numpy,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "pflops_total": self.training_flops_total / 1e15,
            "pflops_non_embedding": self.non_emb_training_flops_total / 1e15,
            "tflops_per_second": flops_this_step / step_time / 1e12,
            "step_time_seconds": step_time,
            "num_tokens": self.num_tokens_seen,
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
        return {"loss": loss}

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def num_non_embedding_parameters(self) -> int:
        return self.model.num_non_embedding_parameters()

    def get_training_pflops(self) -> float:
        return self.training_flops_total / 1e15

    def get_non_emb_training_pflops(self) -> float:
        return self.non_emb_training_flops_total / 1e15

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

    if config.training_config.training_steps_per_epoch is None:
        # Default: chinchilla-optimal amount of data, which is 20 tokens per parameter
        num_parameters = model.num_non_embedding_parameters()
        num_tokens_per_step = config.batch_size * config.sequence_length
        chinchilla_optimal_steps = int(20 * num_parameters / num_tokens_per_step)
        # update frozen dataclass
        config = dataclasses.replace(
            config,
            training_config=dataclasses.replace(
                config.training_config,
                training_steps_per_epoch=chinchilla_optimal_steps,
            ),
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
                training_steps_per_epoch=None,
                seed=42,
            ),
            eval_config=EvalConfig(
                every_n_steps=50,
                steps=10,
                batch_size=64,
                sequence_length=512,
            ),
            model_config=TransformerConfig(
                num_layers=15,
                num_heads=12,
                num_heads_kv=6,
                head_dim=64,
                mlp_inner_size=3072,
                embedding_size=768,
            ),
        )
        losses = train_language_model(config, neptune_run=neptune_run)
        print(f"Losses: {losses}")
    finally:
        neptune_run.stop()


if __name__ == "__main__":

    # command line args, including name
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_neptune", action="store_true", default=False)
    args = parser.parse_args()

    run(use_neptune=not args.no_neptune)
