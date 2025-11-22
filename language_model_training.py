"""
Language model training.
"""

import argparse
import dataclasses
import os
import time
import torch.cuda.nvtx as nvtx
import neptune_lib

from training_basics import (
    TrainingConfig,
    ShuffleBuffer,
    TrainingState,
    Metrics,
)
from language_model_basics import (
    LMData,
    LanguageModelTrainingConfig,
    EvalConfig,
)
from training_loop import train
import torch
import torch.nn as nn
import torch.optim as optim

import stackv2_dataloader
import slimpajama_dataloader
import prng
import language_model_basics
import transformer
import model_configs.chinchilla  # noqa: F401
import cross_entropy

# import bf16_fused_adam
import optimi  # for 16-bit optimizers

# cache for torch.compile to improve startup times.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_cache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

torch.set_float32_matmul_precision("high")  # enable use TF32 to enable tensor cores

assert torch.cuda.is_available(), "CUDA required"


def get_auto_learning_rate(num_parameters: int) -> float:
    """Automatically select learning rate based on model size (number of parameters).

    This uses heuristics based on Chinchilla scaling laws.

    Args:
        num_parameters: Number of non-embedding parameters in the model

    Returns:
        Appropriate learning rate for the model size
    """
    # Convert to millions for easier comparison
    size_m = num_parameters / 1_000_000

    if size_m <= 100:
        return 0.002
    elif size_m <= 200:
        return 0.0015
    elif size_m <= 300:
        return 0.002
    elif size_m <= 500:
        return 0.001
    elif size_m <= 1000:
        return 0.0007
    elif size_m <= 1500:
        return 0.0005
    else:
        return 0.0003


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


class LanguageModelTrainingState(TrainingState[LMData]):
    """Training state for language model"""

    def __init__(
        self,
        model: language_model_basics.LanguageModel,
        config: LanguageModelTrainingConfig,
    ):
        self.model = model
        self.config = config

        with prng.PRNG(config.seed + 345345):
            # self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
            # https://optimi.benjaminwarner.dev/kahan_summation/
            self.optimizer = optimi.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                # eps 1e-8 is the default in pytorch AdamW, and
                # 1e-6 in optimi.
                eps=config.adam_eps,
                betas=config.adam_betas,
                weight_decay=config.weight_decay,
            )

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

    def step(self, data: LMData) -> Metrics:
        inputs = torch.tensor(data.inputs, dtype=torch.int32)
        targets = torch.tensor(data.targets, dtype=torch.long)
        loss_mask = torch.tensor(data.loss_mask, dtype=torch.float32)
        assert inputs.shape == targets.shape == loss_mask.shape
        assert inputs.shape == (
            self.config.batch_size,
            self.config.sequence_length,
        )

        start = time.time()
        targets = torch.where(
            loss_mask == 0.0,
            cross_entropy.cross_entropy_ignore_index,
            targets,
        )

        with nvtx.range("train_step", color="blue"):
            torch.compiler.cudagraph_mark_step_begin()
            loss = self.model.compute_loss(inputs, targets)

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
        self.num_tokens_seen += inputs.numel()

        return {
            "loss": loss_numpy,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "pflops_total": self.training_flops_total / 1e15,
            "tflops_per_second": flops_this_step / step_time / 1e12,
            "step_time_seconds": step_time,
            "num_tokens": self.num_tokens_seen,
        }

    @torch.no_grad()
    def eval(self, data: LMData) -> Metrics:
        inputs = torch.tensor(data.inputs, dtype=torch.int32)
        targets = torch.tensor(data.targets, dtype=torch.long)
        loss_mask = torch.tensor(data.loss_mask, dtype=torch.float32)
        assert inputs.shape == (
            self.config.eval_config.batch_size,
            self.config.eval_config.sequence_length,
        )
        targets = torch.where(
            loss_mask == 0.0,
            cross_entropy.cross_entropy_ignore_index,
            targets,
        )
        loss = self.model.compute_loss(inputs, targets)
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

    # Auto-select learning rate if not provided
    if config.learning_rate is None:
        num_parameters = model.num_non_embedding_parameters()
        auto_lr = get_auto_learning_rate(num_parameters)
        print(
            f"Auto-selecting learning rate based on model size ({num_parameters:,} params): {auto_lr}"
        )
        config = dataclasses.replace(config, learning_rate=auto_lr)

    if config.training_config.training_steps_per_epoch is None:
        # Chinchilla-optimal amount of data, which is 20 tokens per parameter
        num_parameters = model.num_non_embedding_parameters()
        num_tokens_per_step = config.batch_size * config.sequence_length
        chinchilla_optimal_steps = int(
            20 * num_parameters * config.chinchilla_factor / num_tokens_per_step
        )
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
        train_dataset = (
            slimpajama_dataloader.create_slimpajama_dataloader_in_separate_process(
                config
            )
        )
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
    config: LanguageModelTrainingConfig,
    description: str,
    run_name: str | None = None,
    use_neptune: bool = False,
    gpu_id: int | None = None,
    neptune_tags: list[str] = [],
):
    # Add device detection at the top of your training function
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
    else:
        device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    torch.set_default_device(device)

    neptune_run = neptune_lib.NeptuneRunWrapper(
        use_neptune,
        description,
        run_name,
        tags=neptune_tags,
    )
    try:
        losses = train_language_model(config, neptune_run=neptune_run)
        print(f"Losses: {losses}")
    finally:
        neptune_run.stop()


def get_model_config(
    model_config_str: str,
    profile_only: bool = False,
):
    model_config = transformer.transformer_config_registry.get(model_config_str)
    return LanguageModelTrainingConfig(
        name=model_config_str,
        vocab_size=100277,
        learning_rate=None,  # Auto-select based on model size
        batch_size=192,
        training_config=TrainingConfig(
            num_epochs=1,
            training_steps_per_epoch=(
                None if not profile_only else 10
            ),  # None defaults to Chinchilla
            seed=42,
        ),
        model_config=model_config,
    )


if __name__ == "__main__":

    """Example usage:

    python language_model_training.py --no_neptune --model_config chinchilla-44m --profile_only
    """

    # command line args, including name
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_neptune", action="store_true", default=False)
    parser.add_argument("--model_config", type=str, default="chinchilla-44m")
    parser.add_argument("--profile_only", action="store_true", default=False)
    parser.add_argument("--description", "-d", type=str, default=None)
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--gpu_id", "-g", type=int, default=None)
    parser.add_argument("--neptune_tags", type=str, nargs="+", default=[])
    args = parser.parse_args()

    config = get_model_config(args.model_config, args.profile_only)
    if args.profile_only:
        config = dataclasses.replace(
            config,
            training_config=dataclasses.replace(
                config.training_config,
                training_steps_per_epoch=10,
                train_metrics_every_n_steps=1,
            ),
        )
    run(
        config=config,
        use_neptune=not args.no_neptune and not args.profile_only,
        description=args.description,
        run_name=args.name,
        gpu_id=args.gpu_id,
        neptune_tags=args.neptune_tags,
    )
