"""Small CLI helper to run lm-evaluation-harness on a littletrainingloop checkpoint.

This wraps `lm_eval_wrapper.evaluate_checkpoint` so you can do:

    python run_lm_eval.py \
        --checkpoint_path checkpoints/my_run/epoch_1_step_1000.pt \
        --tasks hellaswag,arc_easy \
        --limit 100
"""

import argparse
import json
from pathlib import Path

from lm_eval_wrapper import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval on a training checkpoint.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to .pt training checkpoint")
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated list of lm-eval task names, e.g. 'hellaswag,arc_easy'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples per task (for quick smoke tests)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run evaluation on (e.g. 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to write full evaluation results as JSON.",
    )

    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("No valid tasks provided. Use e.g. --tasks hellaswag,arc_easy")

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        tasks=tasks,
        limit=args.limit,
        device=args.device,
    )

    # `results` is the dict returned by lm_eval.simple_evaluate.
    metrics = results.get("results", results)

    # Pretty-print per-task metrics to stdout.
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Optionally write full results dictionary to disk.
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote full results to {output_path}")


if __name__ == "__main__":
    main()
