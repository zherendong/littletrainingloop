"""
Generic training loop to understand the minimal requirements for a training loop.

Design goals:
- no torch, numpy, etc
- generate a clean list of functions to implement
"""

from typing import Sequence

from training_basics import (
    TrainingConfig,
    DataProvider,
    TrainingState,
    Metrics,
    D,
)


def print_metrics(metrics: Metrics) -> None:
    """Print metrics sorted by name"""
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")


def do_eval(
    config: TrainingConfig,
    state: TrainingState[D],
    eval_data_providers: Sequence[DataProvider[D]],
    epoch: int | None = None,
    step: int | None = None,
) -> Metrics:
    """Evaluate the model"""
    print(f"Eval metrics ({epoch=}, {step=}):")
    for eval_data_provider in eval_data_providers:
        print(f"  {eval_data_provider.get_name()}:")
        for idx, data in enumerate(eval_data_provider.generate()):
            if idx > config.eval_steps:
                break
            metrics = state.eval(data)
            print_metrics(metrics)


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
    eval_data_providers: Sequence[DataProvider[D]] = (),
):
    """Training loop using configuration object"""

    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        for idx, data in enumerate(data_provider.generate()):
            if idx % config.eval_every_n_steps == 0:
                do_eval(config, state, eval_data_providers, epoch, idx)
            if idx >= config.training_steps_per_epoch:
                break
            metrics = state.step(data)
            print(f"Step {idx}:")
            print_metrics(metrics)

        print(f"Epoch {epoch + 1} completed.")
        do_eval(config, state, eval_data_providers, epoch, idx)

    print("-" * 50)
    print("Training completed!")
    return
