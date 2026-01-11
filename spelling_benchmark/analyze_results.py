"""
Analyze spelling benchmark results with stratified metrics.

Can work in two modes:
1. From lm-eval results file (limited to samples saved in the file)
2. Direct inference mode (runs model on full benchmark for complete analysis)

Usage:
    # From lm-eval results
    python -m spelling_benchmark.analyze_results --results_file lm_eval_results/ckpt_spelling_bee.jsonl

    # Direct inference (full analysis)
    python -m spelling_benchmark.analyze_results --checkpoint checkpoints/model.pt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SampleResult:
    """Result for a single sample."""
    doc_id: int
    input_text: str
    target: str
    prediction: str
    correct: bool
    task_type: str
    is_single_token: bool
    num_tokens: int
    word: str


def normalize_prediction(pred: str) -> str:
    """Normalize prediction for comparison."""
    # Strip whitespace, take first line, lowercase
    pred = pred.strip().split('\n')[0].strip().lower()
    return pred


def load_results_from_lm_eval(results_file: Path) -> list[SampleResult]:
    """Load results from lm-eval output file."""
    with open(results_file) as f:
        data = json.load(f)

    if not data.get("success"):
        raise ValueError(f"Evaluation was not successful: {data.get('error')}")

    samples = data.get("samples", [])
    if not samples:
        raise ValueError("No samples found in results file")

    results = []
    for i, sample in enumerate(samples):
        doc = sample.get("doc", {})

        # Get the model's response
        # For generate_until, resps is a list of (response_text, is_greedy) tuples
        resps = sample.get("resps", [[]])
        if resps and resps[0]:
            # Take first response, first element of tuple
            pred = resps[0][0] if isinstance(resps[0], (list, tuple)) else resps[0]
        else:
            pred = ""

        target = doc.get("target", "")
        pred_normalized = normalize_prediction(pred)
        target_normalized = target.strip().lower()

        metadata = doc.get("metadata", {})

        results.append(SampleResult(
            doc_id=sample.get("doc_id", i),
            input_text=doc.get("input", ""),
            target=target,
            prediction=pred,
            correct=(pred_normalized == target_normalized),
            task_type=doc.get("task_type", "unknown"),
            is_single_token=metadata.get("is_single_token", False),
            num_tokens=metadata.get("num_tokens", 0),
            word=metadata.get("word", ""),
        ))

    return results


def run_direct_inference(checkpoint_path: Path, benchmark_path: Path, device: str = "cuda") -> list[SampleResult]:
    """Run inference directly on the benchmark using lm_eval_wrapper."""
    import torch
    torch.set_default_device(device)
    import checkpointing
    import lm_eval_wrapper

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = checkpointing.load_model_from_training_checkpoint(checkpoint_path, device=device)

    # Create wrapper (same as used by lm-eval)
    wrapper = lm_eval_wrapper.LittleTrainingLoopWrapper(
        model=checkpoint.model,
        config=checkpoint.config,
        device=device,
        batch_size=1,
        generate_until_max_length=32,
    )

    # Load benchmark
    samples = []
    with open(benchmark_path) as f:
        for line in f:
            samples.append(json.loads(line))

    results = []
    print(f"Running inference on {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(samples)}")

        input_text = sample["input"]
        target = sample["target"]

        # Generate using the same method as lm-eval (greedy, stop at newline)
        pred = wrapper.infer(input_text, until=["\n"])

        # Check if prediction matches
        pred_normalized = normalize_prediction(pred)
        target_normalized = target.strip().lower()

        metadata = sample.get("metadata", {})

        results.append(SampleResult(
            doc_id=i,
            input_text=input_text,
            target=target,
            prediction=pred,
            correct=(pred_normalized == target_normalized),
            task_type=sample.get("task_type", "unknown"),
            is_single_token=metadata.get("is_single_token", False),
            num_tokens=metadata.get("num_tokens", 0),
            word=metadata.get("word", ""),
        ))

    return results


def compute_accuracy(results: list[SampleResult]) -> tuple[int, int, float]:
    """Compute accuracy for a list of results."""
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    acc = correct / total if total > 0 else 0.0
    return correct, total, acc


def generate_report(results: list[SampleResult]) -> str:
    """Generate a detailed analysis report."""
    lines = []
    lines.append("=" * 60)
    lines.append("SPELLING BENCHMARK ANALYSIS")
    lines.append("=" * 60)

    # Overall accuracy
    correct, total, acc = compute_accuracy(results)
    lines.append(f"\nOverall: {acc:.1%} ({correct}/{total})")

    # By token count
    lines.append("\n" + "-" * 40)
    lines.append("By Token Count:")
    lines.append("-" * 40)

    single_token = [r for r in results if r.is_single_token]
    multi_token = [r for r in results if not r.is_single_token]

    if single_token:
        c, t, a = compute_accuracy(single_token)
        lines.append(f"  Single-token: {a:.1%} ({c}/{t})")
    if multi_token:
        c, t, a = compute_accuracy(multi_token)
        lines.append(f"  Multi-token:  {a:.1%} ({c}/{t})")

    # By task type
    lines.append("\n" + "-" * 40)
    lines.append("By Task Type:")
    lines.append("-" * 40)

    by_task = defaultdict(list)
    for r in results:
        by_task[r.task_type].append(r)

    for task_type in ["count", "index", "reverse"]:
        if task_type in by_task:
            c, t, a = compute_accuracy(by_task[task_type])
            lines.append(f"  {task_type.capitalize():8s}: {a:.1%} ({c}/{t})")

    # By task type × token count
    lines.append("\n" + "-" * 40)
    lines.append("By Task Type × Token Count:")
    lines.append("-" * 40)

    for task_type in ["count", "index", "reverse"]:
        if task_type not in by_task:
            continue
        task_results = by_task[task_type]
        single = [r for r in task_results if r.is_single_token]
        multi = [r for r in task_results if not r.is_single_token]

        if single:
            c, t, a = compute_accuracy(single)
            lines.append(f"  {task_type.capitalize():8s} (single): {a:.1%} ({c}/{t})")
        if multi:
            c, t, a = compute_accuracy(multi)
            lines.append(f"  {task_type.capitalize():8s} (multi):  {a:.1%} ({c}/{t})")

    # Error analysis - show some failures
    lines.append("\n" + "-" * 40)
    lines.append("Sample Errors (first 10):")
    lines.append("-" * 40)

    errors = [r for r in results if not r.correct][:10]
    for r in errors:
        lines.append(f"  [{r.task_type}] '{r.word}'")
        lines.append(f"    Target: '{r.target}' | Predicted: '{normalize_prediction(r.prediction)}'")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def save_detailed_results(results: list[SampleResult], output_path: Path):
    """Save per-sample results to JSONL for further analysis."""
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "doc_id": r.doc_id,
                "input": r.input_text,
                "target": r.target,
                "prediction": r.prediction,
                "correct": r.correct,
                "task_type": r.task_type,
                "is_single_token": r.is_single_token,
                "num_tokens": r.num_tokens,
                "word": r.word,
            }) + "\n")
    print(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze spelling benchmark results")
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to lm-eval results JSONL file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for direct inference",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="spelling_benchmark/spelling_bee.jsonl",
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save detailed per-sample results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    args = parser.parse_args()

    if args.results_file:
        print(f"Loading results from: {args.results_file}")
        results = load_results_from_lm_eval(Path(args.results_file))
        print(f"Loaded {len(results)} samples")
        if len(results) < 1000:
            print(f"WARNING: Only {len(results)} samples loaded. "
                  "lm-eval may have truncated samples. "
                  "Use --checkpoint for full analysis.")
    elif args.checkpoint:
        results = run_direct_inference(
            Path(args.checkpoint),
            Path(args.benchmark),
            args.device,
        )
    else:
        parser.error("Must specify either --results_file or --checkpoint")

    # Generate and print report
    report = generate_report(results)
    print(report)

    # Optionally save detailed results
    if args.output:
        save_detailed_results(results, Path(args.output))


if __name__ == "__main__":
    main()
