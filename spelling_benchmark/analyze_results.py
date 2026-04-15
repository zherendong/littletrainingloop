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


def format_delta(baseline_acc: float, compare_acc: float) -> str:
    """Format accuracy delta with color indicator."""
    delta = compare_acc - baseline_acc
    if delta > 0:
        return f"+{delta:.1%}"
    elif delta < 0:
        return f"{delta:.1%}"
    else:
        return "0.0%"


def generate_comparison_report(
    baseline_results: list[SampleResult],
    compare_results: list[SampleResult],
    baseline_name: str = "Baseline",
    compare_name: str = "Spelling Bee",
) -> str:
    """Generate a side-by-side comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SPELLING BENCHMARK COMPARISON")
    lines.append("=" * 70)
    lines.append(f"\n  {baseline_name:20s} vs {compare_name:20s}")

    # Overall accuracy
    _, _, base_acc = compute_accuracy(baseline_results)
    _, _, comp_acc = compute_accuracy(compare_results)
    delta = format_delta(base_acc, comp_acc)

    lines.append("\n" + "-" * 70)
    lines.append(f"{'Metric':<30s} {baseline_name:>12s} {compare_name:>12s} {'Delta':>10s}")
    lines.append("-" * 70)
    lines.append(f"{'Overall':<30s} {base_acc:>11.1%} {comp_acc:>12.1%} {delta:>10s}")

    # By token count
    lines.append("\n" + "-" * 70)
    lines.append("By Token Count:")
    lines.append("-" * 70)

    for token_type, filter_fn in [
        ("Single-token", lambda r: r.is_single_token),
        ("Multi-token", lambda r: not r.is_single_token),
    ]:
        base_filtered = [r for r in baseline_results if filter_fn(r)]
        comp_filtered = [r for r in compare_results if filter_fn(r)]

        if base_filtered and comp_filtered:
            _, _, base_acc = compute_accuracy(base_filtered)
            _, _, comp_acc = compute_accuracy(comp_filtered)
            delta = format_delta(base_acc, comp_acc)
            lines.append(f"  {token_type:<28s} {base_acc:>11.1%} {comp_acc:>12.1%} {delta:>10s}")

    # By task type
    lines.append("\n" + "-" * 70)
    lines.append("By Task Type:")
    lines.append("-" * 70)

    for task_type in ["count", "index", "reverse"]:
        base_filtered = [r for r in baseline_results if r.task_type == task_type]
        comp_filtered = [r for r in compare_results if r.task_type == task_type]

        if base_filtered and comp_filtered:
            _, _, base_acc = compute_accuracy(base_filtered)
            _, _, comp_acc = compute_accuracy(comp_filtered)
            delta = format_delta(base_acc, comp_acc)
            lines.append(f"  {task_type.capitalize():<28s} {base_acc:>11.1%} {comp_acc:>12.1%} {delta:>10s}")

    # By task type × token count (the key comparison)
    lines.append("\n" + "-" * 70)
    lines.append("By Task Type × Token Count:")
    lines.append("-" * 70)

    for task_type in ["count", "index", "reverse"]:
        for token_label, filter_fn in [
            ("single", lambda r: r.is_single_token),
            ("multi", lambda r: not r.is_single_token),
        ]:
            base_filtered = [r for r in baseline_results if r.task_type == task_type and filter_fn(r)]
            comp_filtered = [r for r in compare_results if r.task_type == task_type and filter_fn(r)]

            if base_filtered and comp_filtered:
                _, _, base_acc = compute_accuracy(base_filtered)
                _, _, comp_acc = compute_accuracy(comp_filtered)
                delta = format_delta(base_acc, comp_acc)
                label = f"{task_type.capitalize()} ({token_label})"
                lines.append(f"  {label:<28s} {base_acc:>11.1%} {comp_acc:>12.1%} {delta:>10s}")

    # Summary insights
    lines.append("\n" + "=" * 70)
    lines.append("KEY INSIGHTS:")
    lines.append("=" * 70)

    # Compare single vs multi token improvement
    base_single = [r for r in baseline_results if r.is_single_token]
    comp_single = [r for r in compare_results if r.is_single_token]
    base_multi = [r for r in baseline_results if not r.is_single_token]
    comp_multi = [r for r in compare_results if not r.is_single_token]

    if base_single and comp_single and base_multi and comp_multi:
        _, _, base_single_acc = compute_accuracy(base_single)
        _, _, comp_single_acc = compute_accuracy(comp_single)
        _, _, base_multi_acc = compute_accuracy(base_multi)
        _, _, comp_multi_acc = compute_accuracy(comp_multi)

        single_improvement = comp_single_acc - base_single_acc
        multi_improvement = comp_multi_acc - base_multi_acc

        lines.append(f"  Single-token improvement: {single_improvement:+.1%}")
        lines.append(f"  Multi-token improvement:  {multi_improvement:+.1%}")

        if single_improvement > multi_improvement + 0.01:
            lines.append("\n  -> Spelling bee embeddings help MORE on single-token words (as expected)")
        elif multi_improvement > single_improvement + 0.01:
            lines.append("\n  -> Spelling bee embeddings help MORE on multi-token words (unexpected)")
        else:
            lines.append("\n  -> Similar improvement on both token types")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze spelling benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model analysis
  python -m spelling_benchmark.analyze_results --results_file results/baseline.jsonl

  # Compare two models
  python -m spelling_benchmark.analyze_results \\
    --results_file results/baseline.jsonl \\
    --compare results/spelling_bee.jsonl
        """,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Path to lm-eval results JSONL file (baseline for comparison)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to second results file to compare against baseline",
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default="Baseline",
        help="Display name for baseline model (default: Baseline)",
    )
    parser.add_argument(
        "--compare_name",
        type=str,
        default="Spelling Bee",
        help="Display name for comparison model (default: Spelling Bee)",
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

    # Comparison mode
    if args.compare:
        if not args.results_file:
            parser.error("--compare requires --results_file as baseline")

        print(f"Loading baseline results from: {args.results_file}")
        baseline_results = load_results_from_lm_eval(Path(args.results_file))
        print(f"Loaded {len(baseline_results)} baseline samples")

        print(f"Loading comparison results from: {args.compare}")
        compare_results = load_results_from_lm_eval(Path(args.compare))
        print(f"Loaded {len(compare_results)} comparison samples")

        report = generate_comparison_report(
            baseline_results,
            compare_results,
            baseline_name=args.baseline_name,
            compare_name=args.compare_name,
        )
        print(report)
        return

    # Single model analysis
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
