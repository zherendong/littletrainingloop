#!/usr/bin/env python3
"""
Visualize GLU experiment results.

This script loads metrics from JSON files and creates comparison plots.

Usage:
    # Plot all experiments
    python plot_glu_results.py
    
    # Plot specific experiments
    python plot_glu_results.py --experiments c44m_baseline_lr2e-3_w100 c44m_swiglu_lr2e-3_w100
    
    # Plot by variant (baseline, geglu, swiglu)
    python plot_glu_results.py --group-by variant
    
    # Save to file instead of showing
    python plot_glu_results.py --output glu_comparison.png
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from metrics_logger import load_metrics, list_experiments


def parse_experiment_name(name: str) -> dict:
    """
    Parse experiment name to extract metadata.
    
    Example: "c44m_baseline_lr2e-3_w100" ->
        {
            "model": "c44m",
            "variant": "baseline",
            "lr": 0.0015,
            "warmup": 100
        }
    """
    # Pattern: {model}_{variant}_lr{lr}_w{warmup}
    pattern = r"(c\d+m)_(\w+)_lr([\d.e-]+)_w(\d+)"
    match = re.match(pattern, name)
    
    if not match:
        return {"name": name}
    
    model, variant, lr_str, warmup = match.groups()
    
    # Parse learning rate (e.g., "2e-3" -> 0.002)
    lr = float(lr_str.replace("e-", "e-"))
    
    return {
        "name": name,
        "model": model,
        "variant": variant,
        "lr": lr,
        "warmup": int(warmup),
    }


def plot_loss_curves(
    experiments: list[str],
    metric_name: str = "eval/SlimPajama_validation/loss",
    x_axis: str = "step",  # "step" or "tokens"
    output_file: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot loss curves for multiple experiments.
    
    Args:
        experiments: List of experiment names
        metric_name: Which metric to plot
        x_axis: "step" or "tokens" for x-axis
        output_file: If provided, save to file instead of showing
        title: Plot title
    """
    plt.figure(figsize=(12, 7))
    
    for exp_name in experiments:
        try:
            data = load_metrics(exp_name)
        except FileNotFoundError:
            print(f"⚠️  Experiment not found: {exp_name}")
            continue
        
        metrics = data["metrics"]
        
        if metric_name not in metrics:
            print(f"⚠️  Metric '{metric_name}' not found in {exp_name}")
            continue
        
        metric_data = metrics[metric_name]
        steps = [m["step"] for m in metric_data]
        values = [m["value"] for m in metric_data]
        
        # Get x-axis data
        if x_axis == "tokens":
            # Try to get num_tokens metric
            tokens_metric = "eval/SlimPajama_validation/num_tokens"
            if tokens_metric in metrics:
                x_data = [m["value"] for m in metrics[tokens_metric]]
                x_label = "Tokens"
            else:
                print(f"⚠️  Tokens metric not found for {exp_name}, using steps")
                x_data = steps
                x_label = "Steps"
        else:
            x_data = steps
            x_label = "Steps"
        
        # Parse experiment name for label
        parsed = parse_experiment_name(exp_name)
        if "variant" in parsed:
            label = f"{parsed['variant']} (LR={parsed['lr']:.0e})"
        else:
            label = exp_name
        
        plt.plot(x_data, values, marker='o', label=label, linewidth=2, markersize=4)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title(title or "GLU Variants Comparison - Validation Loss", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
    else:
        plt.show()


def plot_grouped_comparison(
    experiments: list[str],
    group_by: str = "variant",  # "variant" or "lr"
    metric_name: str = "eval/SlimPajama_validation/loss",
    output_file: Optional[str] = None,
):
    """
    Create subplots grouped by variant or learning rate.
    
    Args:
        experiments: List of experiment names
        group_by: "variant" or "lr"
        metric_name: Which metric to plot
        output_file: If provided, save to file instead of showing
    """
    # Group experiments
    groups = {}
    for exp_name in experiments:
        parsed = parse_experiment_name(exp_name)
        if group_by not in parsed:
            continue
        
        group_key = parsed[group_by]
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(exp_name)
    
    # Create subplots
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), sharey=True)
    
    if n_groups == 1:
        axes = [axes]
    
    for ax, (group_key, group_exps) in zip(axes, sorted(groups.items())):
        for exp_name in group_exps:
            try:
                data = load_metrics(exp_name)
            except FileNotFoundError:
                continue
            
            metrics = data["metrics"]
            if metric_name not in metrics:
                continue
            
            metric_data = metrics[metric_name]
            steps = [m["step"] for m in metric_data]
            values = [m["value"] for m in metric_data]
            
            parsed = parse_experiment_name(exp_name)
            if group_by == "variant":
                label = f"LR={parsed.get('lr', '?'):.0e}"
            else:
                label = parsed.get('variant', exp_name)
            
            ax.plot(steps, values, marker='o', label=label, linewidth=2, markersize=4)
        
        ax.set_xlabel("Steps", fontsize=11)
        ax.set_title(f"{group_by.capitalize()}: {group_key}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("Validation Loss", fontsize=11)
    fig.suptitle("GLU Variants Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
    else:
        plt.show()


def print_summary_table(experiments: list[str]):
    """Print a summary table of final losses."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Experiment':<40} {'Variant':<12} {'LR':<10} {'Final Loss':<12} {'Min Loss':<12}")
    print("-"*80)
    
    results = []
    for exp_name in experiments:
        try:
            data = load_metrics(exp_name)
        except FileNotFoundError:
            continue
        
        summary = data["summary"]
        metric_name = "eval/SlimPajama_validation/loss"
        
        if metric_name not in summary:
            continue
        
        final_loss = summary[metric_name]["final"]
        min_loss = summary[metric_name]["min"]
        
        parsed = parse_experiment_name(exp_name)
        variant = parsed.get("variant", "?")
        lr = parsed.get("lr", 0)
        
        results.append({
            "name": exp_name,
            "variant": variant,
            "lr": lr,
            "final_loss": final_loss,
            "min_loss": min_loss,
        })
        
        print(f"{exp_name:<40} {variant:<12} {lr:<10.0e} {final_loss:<12.4f} {min_loss:<12.4f}")
    
    print("="*80)
    
    # Find best per variant
    if results:
        print("\nBEST RESULTS PER VARIANT:")
        print("-"*80)
        variants = {}
        for r in results:
            v = r["variant"]
            if v not in variants or r["final_loss"] < variants[v]["final_loss"]:
                variants[v] = r
        
        for variant, best in sorted(variants.items()):
            print(f"  {variant:<12}: {best['final_loss']:.4f} (LR={best['lr']:.0e}, {best['name']})")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize GLU experiment results")
    
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="+",
        help="Specific experiments to plot (default: all)",
    )
    
    parser.add_argument(
        "--metric",
        "-m",
        default="eval/SlimPajama_validation/loss",
        help="Metric to plot (default: eval/SlimPajama_validation/loss)",
    )
    
    parser.add_argument(
        "--x-axis",
        choices=["step", "tokens"],
        default="step",
        help="X-axis: steps or tokens (default: step)",
    )
    
    parser.add_argument(
        "--group-by",
        choices=["variant", "lr", "none"],
        default="none",
        help="Group plots by variant or learning rate (default: none)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: show plot)",
    )
    
    parser.add_argument(
        "--title",
        "-t",
        help="Plot title",
    )
    
    parser.add_argument(
        "--base-dir",
        default="experiments/metrics",
        help="Base directory for metrics (default: experiments/metrics)",
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary table, no plots",
    )
    
    args = parser.parse_args()
    
    # Get experiments to plot
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = list_experiments(args.base_dir)
        if not experiments:
            print(f"No experiments found in {args.base_dir}")
            return
        print(f"Found {len(experiments)} experiments")
    
    # Print summary table
    print_summary_table(experiments)
    
    if args.summary_only:
        return
    
    # Create plots
    if args.group_by == "none":
        plot_loss_curves(
            experiments,
            metric_name=args.metric,
            x_axis=args.x_axis,
            output_file=args.output,
            title=args.title,
        )
    else:
        plot_grouped_comparison(
            experiments,
            group_by=args.group_by,
            metric_name=args.metric,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()

