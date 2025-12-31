"""
Generic training loop to understand the minimal requirements for a training loop.

Design goals:
- no torch, numpy, etc
- generate a clean list of functions to implement
"""

import time
from typing import Sequence

import torch

from training_basics import (
    TrainingConfig,
    EvalConfig,
    DataProvider,
    TrainingState,
    MetricItem,
    D,
)
import neptune_lib


def mem_gb():
    s = torch.cuda.memory_stats()
    alloc = s["allocated_bytes.all.current"] / 1e9
    peak = s["allocated_bytes.all.peak"] / 1e9
    resv = s["reserved_bytes.all.current"] / 1e9
    frag = resv - alloc
    print(f"beginning of step: {alloc=:.2f}, {peak=:.2f}, {resv=:.2f}, {frag=:.2f}")
    torch.cuda.reset_peak_memory_stats()
    return alloc, peak, resv, frag


def record_metrics(
    metrics: dict[str, MetricItem],
    metric_path: str,
    step: int,
    neptune_run: neptune_lib.NeptuneRunWrapper,
):
    """Record metrics with custom X-axis values.

    Handles both training and evaluation metrics. Iterates through MetricItem instances
    and logs them to Neptune. For each metric, uses the custom x_axis if provided,
    otherwise uses the passed step.

    Args:
        metrics: Dict mapping metric names to MetricItem instances.
                 Each MetricItem contains a value and optional custom x_axis.
        metric_path: Path for the metric (e.g., "eval/validation", "eval/lm_eval_harness").
        step: Step to use for metrics where MetricItem.x_axis is None.
        neptune_run: Neptune run for logging.
    """
    print(f"{metric_path}:")
    for key, item in sorted(metrics.items()):
        print(f"  {key}: {item.value:.6f}")
    for key, item in metrics.items():
        x = step if item.x_axis is None else item.x_axis
        neptune_run[f"{metric_path}/{key}"].append(item.value, step=x)


def validation(
    *,
    config: EvalConfig,
    state: TrainingState[D],
    eval_data_providers: Sequence[DataProvider[D]],
    epoch: int,
    step: int,
    neptune_run: neptune_lib.NeptuneRunWrapper,
):
    """Evaluate the model"""
    print(f"Eval metrics ({epoch=}, {step=}):")
    for eval_data_provider in eval_data_providers:
        start_time = time.time()
        print(f"  {eval_data_provider.get_name()}:")
        final_metrics = state.validation_loss(
            eval_data_provider.generate(), config.steps
        )
        name = eval_data_provider.get_name()
        record_metrics(
            metrics=final_metrics,
            metric_path=f"eval/{name}",
            step=step,
            neptune_run=neptune_run,
        )
        print(f"Eval {name} completed in {time.time() - start_time:.2f}s")


def lm_eval(
    step: int,
    state: TrainingState[D],
    neptune_run: neptune_lib.NeptuneRunWrapper,
):
    """Evaluate the model"""
    metrics = state.evaluate()
    record_metrics(
        metrics=metrics,
        metric_path="eval/lm_eval_harness",
        step=step,
        neptune_run=neptune_run,
    )


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
    eval_config: EvalConfig,
    neptune_run: neptune_lib.NeptuneRunWrapper,
    eval_data_providers: Sequence[DataProvider[D]] = (),
):
    """Training loop using configuration object"""

    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        idx = 0
        for idx, data in enumerate(data_provider.generate()):
            if idx % eval_config.every_n_steps == 0:
                validation(
                    config=eval_config,
                    state=state,
                    eval_data_providers=eval_data_providers,
                    epoch=epoch,
                    step=idx,
                    neptune_run=neptune_run,
                )
            if (
                eval_config.full_eval_every_n_steps is not None
                and idx % eval_config.full_eval_every_n_steps == 0
            ):
                lm_eval(idx, state, neptune_run=neptune_run)

            if (
                config.training_steps_per_epoch
                and idx >= config.training_steps_per_epoch
            ):
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
            if idx % config.train_metrics_every_n_steps == 0 or idx == 10:
                mem_gb()
                record_metrics(
                    metrics=metrics,
                    metric_path="train",
                    step=idx,
                    neptune_run=neptune_run,
                )

        print(f"Epoch {epoch + 1} completed.")
        validation(
            config=eval_config,
            state=state,
            eval_data_providers=eval_data_providers,
            epoch=epoch,
            step=idx,
            neptune_run=neptune_run,
        )

        if config.checkpoint_path is not None:
            print(f"Saving checkpoint to {config.checkpoint_path}")
            state.save_checkpoint(
                config.checkpoint_path, neptune_run.get_run_id(), idx, epoch
            )
        if eval_config.full_eval_every_n_steps is not None:
            lm_eval(idx, state, neptune_run=neptune_run)

    print("-" * 50)
    print("Training completed!")
    return
