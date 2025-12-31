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
    Metrics,
    D,
)
import neptune_lib


def process_metrics(metrics: Metrics, neptune_run, step=None, mode="train") -> None:
    """Print metrics sorted by name"""
    print(f"{mode} step {step}:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.6f}")
    for name, value in sorted(metrics.items()):
        neptune_run[f"{mode}/{name}"].append(value, step=step)


def mem_gb():
    s = torch.cuda.memory_stats()
    alloc = s["allocated_bytes.all.current"] / 1e9
    peak = s["allocated_bytes.all.peak"] / 1e9
    resv = s["reserved_bytes.all.current"] / 1e9
    frag = resv - alloc
    print(f"beginning of step: {alloc=:.2f}, {peak=:.2f}, {resv=:.2f}, {frag=:.2f}")
    torch.cuda.reset_peak_memory_stats()
    return alloc, peak, resv, frag


def record_eval(
    state: TrainingState[D],
    metrics: Metrics,
    step: int | None = None,
    neptune_run=neptune_lib.NullNeptuneRun(),
    eval_name: str = "",
):
    """Record evaluation metrics"""
    for step_val, step_name in [
        (step, ""),
        (
            state.get_training_tokens_seen(),
            "/num_tokens",
        ),  # TODO: replace the "/" to "_vs_" so that it doesn't collide with the existing naming pattern
        (state.get_training_pflops(), "/pflops"),
    ]:
        process_metrics(
            metrics,
            neptune_run=neptune_run,
            step=step_val,
            mode=f"eval/{eval_name}{step_name}",
        )


def validation(
    config: EvalConfig,
    state: TrainingState[D],
    eval_data_providers: Sequence[DataProvider[D]],
    epoch: int | None = None,
    step: int | None = None,
    neptune_run=neptune_lib.NullNeptuneRun(),
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
        record_eval(
            state=state,
            metrics=final_metrics,
            step=step,
            neptune_run=neptune_run,
            eval_name=name,
        )
        print(f"Eval {name} completed in {time.time() - start_time:.2f}s")


def lm_eval(
    step: int,
    state: TrainingState[D],
    neptune_run=neptune_lib.NullNeptuneRun(),
):
    """Evaluate the model"""
    metrics = state.evaluate()
    record_eval(
        state=state,
        metrics=metrics,
        step=step,
        neptune_run=neptune_run,
        eval_name="lm_eval_harness",
    )


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
    eval_config: EvalConfig,
    eval_data_providers: Sequence[DataProvider[D]] = (),
    neptune_run=neptune_lib.NullNeptuneRun(),
):
    """Training loop using configuration object"""

    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        idx = 0
        for idx, data in enumerate(data_provider.generate()):
            if idx % eval_config.every_n_steps == 0:
                validation(
                    eval_config,
                    state,
                    eval_data_providers,
                    epoch,
                    idx,
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
                process_metrics(
                    metrics, neptune_run=neptune_run, step=idx, mode="train"
                )

        print(f"Epoch {epoch + 1} completed.")
        validation(
            eval_config,
            state,
            eval_data_providers,
            epoch,
            idx,
            neptune_run=neptune_run,
        )
        if eval_config.full_eval_every_n_steps is not None:
            lm_eval(idx, state, neptune_run=neptune_run)

        if config.checkpoint_path:
            print(f"Saving checkpoint to {config.checkpoint_path}")
            state.save_checkpoint(
                config.checkpoint_path, neptune_run.get_run_id(), idx, epoch
            )

    print("-" * 50)
    print("Training completed!")
    return
