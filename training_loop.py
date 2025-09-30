"""
Generic training loop to understand the minimal requirements for a training loop.

Design goals:
- no torch, numpy, etc
- generate a clean list of functions to implement
"""

import dataclasses
from collections import defaultdict
from typing import Sequence

from training_basics import (
    TrainingConfig,
    EvalConfig,
    DataProvider,
    TrainingState,
    Metrics,
    D,
)
import metrics


def process_metrics(
    metrics: Metrics, neptune_run=None, step=None, mode="train"
) -> None:
    """Print metrics sorted by name"""
    print(f"{mode} step {step}:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.6f}")
    if neptune_run is not None:
        for name, value in sorted(metrics.items()):
            x_axis = step
            if isinstance(value, tuple):
                value, x_axis = value
            neptune_run[f"{mode}/{name}"].append(value, step=x_axis)


def do_eval(
    config: EvalConfig,
    state: TrainingState[D],
    eval_data_providers: Sequence[DataProvider[D]],
    epoch: int | None = None,
    step: int | None = None,
    neptune_run=None,
) -> Metrics:
    """Evaluate the model"""
    print(f"Eval metrics ({epoch=}, {step=}):")
    for eval_data_provider in eval_data_providers:
        print(f"  {eval_data_provider.get_name()}:")

        metric_aggregators = defaultdict(metrics.ExactMetricsAggregator)
        for idx, data in enumerate(eval_data_provider.generate()):
            if idx > config.steps:
                break
            metrics = state.eval(data)
            for name, value in metrics.items():
                metric_aggregators[name].observe(value)

        # loss = sum(losses) / (len(losses) + 1e-6)
        metrics = {
            name: aggregator.mean() for name, aggregator in metric_aggregators.items()
        }
        process_metrics(
            {"loss": loss},
            neptune_run=neptune_run,
            step=step,
            mode=f"eval/{eval_data_provider.get_name()}",
        )


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
    eval_config: EvalConfig,
    eval_data_providers: Sequence[DataProvider[D]] = (),
    neptune_run=None,
):
    """Training loop using configuration object"""

    if neptune_run is not None:
        neptune_run["num_parameters"] = state.num_parameters()
        neptune_run["num_non_embedding_parameters"] = (
            state.num_non_embedding_parameters()
        )
        neptune_run["config"] = dataclasses.asdict(config)

    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        for idx, data in enumerate(data_provider.generate()):
            if idx % eval_config.every_n_steps == 0:
                do_eval(
                    eval_config,
                    state,
                    eval_data_providers,
                    epoch,
                    idx,
                    neptune_run=neptune_run,
                )
            if idx >= config.training_steps_per_epoch:
                break
            metrics = state.step(data)
            process_metrics(metrics, neptune_run=neptune_run, step=idx, mode="train")

        print(f"Epoch {epoch + 1} completed.")
        do_eval(
            eval_config,
            state,
            eval_data_providers,
            epoch,
            idx,
            neptune_run=neptune_run,
        )

    print("-" * 50)
    print("Training completed!")
    return
