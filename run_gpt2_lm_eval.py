"""Small CLI to run lm-evaluation-harness on a Hugging Face GPT-2 model.

This lets you directly compare your littletrainingloop checkpoints against
standard GPT-2 variants on the same lm-eval tasks.

Example:

    python run_gpt2_lm_eval.py \
        --model_name gpt2 \
        --tasks wikitext,lambada_openai \
        --limit 100 \
        --device cuda

The model weights are downloaded automatically from Hugging Face via
`transformers` through lm-evaluation-harness's built-in HuggingFace backend.
"""

import argparse
import json
from pathlib import Path

from lm_eval import simple_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lm-eval on a Hugging Face GPT-2 model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Hugging Face model id, e.g. 'gpt2', 'gpt2-medium', 'gpt2-large'",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated list of lm-eval task names, e.g. 'wikitext,lambada_openai'",
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
        raise SystemExit("No valid tasks provided. Use e.g. --tasks wikitext,lambada_openai")

    # Use lm-eval's built-in Hugging Face backend (HFLM) via `model="hf"`.
    # `model_args` is passed as a dict here; lm-eval will internally construct
    # the appropriate HFLM wrapper and download the checkpoint from Hugging Face
    # if it is not already cached.
    results = simple_evaluate(
        model="hf",
        model_args={"pretrained": args.model_name},
        tasks=tasks,
        num_fewshot=0,
        limit=args.limit,
        batch_size=1,
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
        output_path.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote full results to {output_path}")


if __name__ == "__main__":
    main()

