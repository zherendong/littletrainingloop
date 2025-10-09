"""
Generic training loop to understand the minimal requirements for a training loop.

Design goals:
- no torch, numpy, etc
- generate a clean list of functions to implement
"""

import time
from collections import defaultdict
from typing import Sequence

import torch

from training_basics import (
    TrainingConfig,
    EvalConfig,
    DataProvider,
    TrainingState,
    Metrics,
    D,
)
import aggregation
import null_neptune


def process_metrics(metrics: Metrics, neptune_run, step=None, mode="train") -> None:
    """Print metrics sorted by name"""
    print(f"{mode} step {step}:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.6f}")
    for name, value in sorted(metrics.items()):
        neptune_run[f"{mode}/{name}"].append(value, step=step)


def do_eval(
    config: EvalConfig,
    state: TrainingState[D],
    eval_data_providers: Sequence[DataProvider[D]],
    epoch: int | None = None,
    step: int | None = None,
    neptune_run=null_neptune.NullNeptuneRun(),
):
    """Evaluate the model"""
    print(f"Eval metrics ({epoch=}, {step=}):")
    for eval_data_provider in eval_data_providers:
        print(f"  {eval_data_provider.get_name()}:")

        metric_aggregators = defaultdict(aggregation.ExactMetricsAggregator)
        for idx, data in enumerate(eval_data_provider.generate()):
            if idx > config.steps:
                break
            metrics = state.eval(data)
            for name, value in metrics.items():
                metric_aggregators[name].observe(value)

        metrics = {
            name: aggregator.mean() for name, aggregator in metric_aggregators.items()
        }
        training_tokens_seen = state.get_training_tokens_seen()
        training_pflops = state.get_training_pflops()
        for step_val, step_name in [
            (step, ""),
            (training_tokens_seen, "/num_tokens"),
            (training_pflops, "/pflops"),
        ]:
            process_metrics(
                metrics,
                neptune_run=neptune_run,
                step=step_val,
                mode=f"eval/{eval_data_provider.get_name()}{step_name}",
            )


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
    eval_config: EvalConfig,
    eval_data_providers: Sequence[DataProvider[D]] = (),
    neptune_run=null_neptune.NullNeptuneRun(),
):
    """Training loop using configuration object"""

    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        idx = 0
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
            if idx == 5:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(torch.cuda.memory_summary())
                # snap = (
                #     torch.cuda.memory_snapshot()
                # )  # JSON-like: blocks, sizes, “active” flags
                # print(snap)

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
