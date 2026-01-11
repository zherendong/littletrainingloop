"""
Robust multi-checkpoint, multi-task evaluation script.

Runs evaluations for each (checkpoint, task) pair independently, catching errors
so that failures don't prevent other evaluations from completing.

Results are saved to lm_eval_results/<checkpoint_name>_<task>.jsonl

Example usage:
    python eval_main.py \
        --checkpoint_paths /path/to/ckpt1.pt /path/to/ckpt2.pt \
        --tasks hellaswag arc_easy arc_challenge
"""

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import lm_eval
from lm_eval.tasks import TaskManager
import torch

import checkpointing
import lm_eval_wrapper

device = "cuda"
torch.set_default_device(device)

RESULTS_DIR = Path("lm_eval_results")


@dataclass
class EvalResult:
    """Result of a single (checkpoint, task) evaluation."""

    checkpoint_path: str
    task: str
    success: bool
    results: dict | None = None
    error: str | None = None
    traceback: str | None = None


def get_checkpoint_name(checkpoint_path: Path) -> str:
    """Extract a clean name from checkpoint path for use in filenames."""
    # Use stem (filename without extension), replace problematic chars
    name = checkpoint_path.stem
    # Also include parent dir name if it's informative
    parent = checkpoint_path.parent.name
    if parent and parent not in (".", ""):
        name = f"{parent}_{name}"
    # Replace any remaining problematic characters
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return name


def save_result(result: EvalResult, output_dir: Path) -> Path:
    """Save evaluation result to a JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = get_checkpoint_name(Path(result.checkpoint_path))
    filename = f"{checkpoint_name}_{result.task}.jsonl"
    filepath = output_dir / filename

    result_dict = {
        "checkpoint_path": result.checkpoint_path,
        "task": result.task,
        "success": result.success,
    }

    if result.success and result.results is not None:
        # Include the task-specific results
        result_dict["results"] = result.results.get("results", {})
        result_dict["configs"] = result.results.get("configs", {})
        result_dict["samples"] = result.results.get("samples", {})
    else:
        result_dict["error"] = result.error
        result_dict["traceback"] = result.traceback

    with open(filepath, "w") as f:
        f.write(json.dumps(result_dict, indent=2, default=str))

    return filepath


def evaluate_single_task(
    wrapper: lm_eval_wrapper.LittleTrainingLoopWrapper,
    task: str,
    num_fewshot: int | None,
    limit: int | None,
    max_samples_log: int | None = 100,
) -> dict:
    """Run evaluation for a single task using an already-loaded model wrapper."""
    print(f"  [lm_eval] Starting evaluation for task: {task}", flush=True)

    # Create task manager with include_path for custom yaml tasks
    # Include spelling_benchmark for the spelling_bee task
    task_manager = TaskManager(include_path="spelling_benchmark")

    results = lm_eval.simple_evaluate(  # type: ignore
        model=wrapper,
        tasks=[task],
        limit=limit,
        num_fewshot=num_fewshot,
        device=wrapper.device,
        max_batch_size=wrapper.batch_size,
        cache_requests=True,
        confirm_run_unsafe_code=True,
        task_manager=task_manager,
    )

    print(f"  [lm_eval] Completed evaluation for task: {task}", flush=True)
    return results


def evaluate_checkpoint(
    checkpoint_path: Path,
    tasks: list[str],
    generate_until_max_length: int | None,
    num_fewshot: int | None,
    limit: int | None,
    output_dir: Path = RESULTS_DIR,
    max_samples_log: int | None = 100,
) -> list[EvalResult]:
    """
    Evaluate a single checkpoint on multiple tasks.

    Loads the checkpoint once, then runs each task independently with error handling.
    """
    results: list[EvalResult] = []
    checkpoint_str = str(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Load checkpoint once
    try:
        checkpoint = checkpointing.load_model_from_training_checkpoint(
            checkpoint_path, device=device
        )
        wrapper = lm_eval_wrapper.LittleTrainingLoopWrapper(
            model=checkpoint.model,
            config=checkpoint.config,
            device=device,
            batch_size=checkpoint.config.eval_config.batch_size,
            generate_until_max_length=generate_until_max_length,
        )
        print(f"Checkpoint loaded successfully.")
    except Exception as e:
        # If checkpoint loading fails, mark all tasks as failed
        tb = traceback.format_exc()
        print(f"ERROR loading checkpoint: {e}")
        print(tb)
        for task in tasks:
            result = EvalResult(
                checkpoint_path=checkpoint_str,
                task=task,
                success=False,
                error=f"Checkpoint loading failed: {str(e)}",
                traceback=tb,
            )
            save_result(result, output_dir)
            results.append(result)
        return results

    # Run each task independently
    for task_name in tasks:
        task = lm_eval_wrapper.get_task_details(task_name)
        print(f"\n  Task: {task}")
        print(f"  {'-'*40}")

        try:
            task_results = evaluate_single_task(
                wrapper=wrapper,
                task=task.name,  # internal name might differ
                num_fewshot=num_fewshot,
                limit=limit,
            )

            # Reduce the number of samples to limit file size
            if "samples" in task_results:
                try:
                    samples = task_results["samples"][task.name]
                    if max_samples_log is not None:
                        samples = samples[:max_samples_log]
                    task_results["samples"] = samples
                except KeyError as e:
                    task_results["samples"] = f"Error: {e}"

            result = EvalResult(
                checkpoint_path=checkpoint_str,
                task=task_name,
                success=True,
                results=task_results,
            )

            # Print summary for this task
            if "results" in task_results and task.name in task_results["results"]:
                task_metrics = task_results["results"][task.name]
                print(f"  Results for {task}:")
                for metric_name, metric_value in task_metrics.items():
                    print(f"    {metric_name}: {metric_value}")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ERROR evaluating task {task}: {e}")
            print(tb)

            result = EvalResult(
                checkpoint_path=checkpoint_str,
                task=task,
                success=False,
                error=str(e),
                traceback=tb,
            )

        # Save result immediately
        filepath = save_result(result, output_dir)
        print(f"  Saved to: {filepath}")
        results.append(result)

    return results


def run_all_evaluations(
    checkpoint_paths: list[Path],
    tasks: list[str],
    generate_until_max_length: int | None,
    num_fewshot: int | None,
    limit: int | None,
    output_dir: Path = RESULTS_DIR,
    max_samples_log: int | None = 100,
) -> list[EvalResult]:
    """
    Run evaluations for all (checkpoint, task) pairs.

    Returns list of all results (successes and failures).
    """
    all_results: list[EvalResult] = []

    total_evals = len(checkpoint_paths) * len(tasks)
    print(f"\nStarting evaluation run:")
    print(f"  Checkpoints: {len(checkpoint_paths)}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Output directory: {output_dir}")

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n[Checkpoint {i+1}/{len(checkpoint_paths)}]")
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            tasks=tasks,
            generate_until_max_length=generate_until_max_length,
            num_fewshot=num_fewshot,
            limit=limit,
            output_dir=output_dir,
            max_samples_log=max_samples_log,
        )
        all_results.extend(results)

    return all_results


def print_summary(results: list[EvalResult]):
    """Print a summary of all evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"\nTotal: {len(results)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")

    if successes:
        print("\n✓ Successful evaluations:")
        for r in successes:
            ckpt_name = get_checkpoint_name(Path(r.checkpoint_path))
            print(f"  {ckpt_name} / {r.task}")

    if failures:
        print("\n✗ Failed evaluations:")
        for r in failures:
            ckpt_name = get_checkpoint_name(Path(r.checkpoint_path))
            print(f"  {ckpt_name} / {r.task}")
            if r.error:
                # Print first line of error
                error_first_line = r.error.split("\n")[0][:80]
                print(f"    Error: {error_first_line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run lm-eval evaluations on multiple checkpoints and tasks."
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to checkpoint files (.pt)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Task names to evaluate (e.g., hellaswag arc_easy)",
    )
    parser.add_argument(
        "--generate_max_length",
        type=int,
        default=None,
        help="Maximum generation length for generative tasks",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples to use (default is set in task definition)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lm_eval_results",
        help="Directory to save results (default: lm_eval_results)",
    )
    parser.add_argument(
        "--max_samples_log",
        type=int,
        default=100,
        help="Max samples to save per task (default: 100, use 0 for unlimited)",
    )

    args = parser.parse_args()

    checkpoint_paths = [Path(p) for p in args.checkpoint_paths]

    # Validate checkpoint paths exist
    for p in checkpoint_paths:
        if not p.exists():
            print(f"Warning: Checkpoint path does not exist: {p}")

    # Validate task names
    invalid_tasks = [t for t in args.tasks if t not in lm_eval_wrapper.available_tasks]
    if invalid_tasks:
        print(f"Error: Unknown task(s): {invalid_tasks}")
        print(f"Available tasks: {list(lm_eval_wrapper.available_tasks.keys())}")
        exit(1)

    # Convert 0 to None for unlimited samples
    max_samples_log = args.max_samples_log if args.max_samples_log > 0 else None

    results = run_all_evaluations(
        checkpoint_paths=checkpoint_paths,
        tasks=args.tasks,
        generate_until_max_length=args.generate_max_length,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_dir=Path(args.output_dir),
        max_samples_log=max_samples_log,
    )

    print_summary(results)
