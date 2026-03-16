"""Plot MLP activation tracking stats from mlp_tracking.py output.

Usage
-----
python plot_mlp_stats.py results/c44m_lr0.004_vanilla_mlp_stats.csv
python plot_mlp_stats.py results/c44m_lr0.004_vanilla_mlp_stats.csv --out_dir results/plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


METRICS = {
    "pct_active": "% active (|act| > threshold)",
    "mean_abs_when_active": "Mean |activation| when active",
    "out_weight_norm": "Output weight column norm",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # block column may not exist in older files
    if "block" not in df.columns:
        df["block"] = 0
    return df


def step_colors(steps):
    """Return a color per step, ranging light->dark over training."""
    cmap = plt.get_cmap("plasma", len(steps))
    return {step: cmap(i) for i, step in enumerate(steps)}


# ── Fig 1: Distribution evolution ─────────────────────────────────────────────

def plot_distribution_evolution(df: pd.DataFrame, out_dir: Path):
    """Histograms of each metric across all neurons+layers, overlaid per step."""
    steps = sorted(df.step.unique())
    colors = step_colors(steps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Distribution of neuron stats over training (all layers)", fontsize=13)

    plot_steps = {col: steps for col in METRICS}
    plot_steps["out_weight_norm"] = [s for s in steps if s != steps[0]]

    for ax, (col, label) in zip(axes, METRICS.items()):
        for step in plot_steps[col]:
            vals = df[df.step == step][col].values
            ax.hist(vals, bins=40, density=True, histtype="step",
                    color=colors[step], alpha=0.8, linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(col)
        if col == "mean_abs_when_active":
            ax.set_xlim(0, 10)
        if col == "out_weight_norm":
            ax.set_ylim(0, 3)

    # shared colorbar for step
    sm = cm.ScalarMappable(cmap="plasma",
                           norm=plt.Normalize(vmin=steps[0], vmax=steps[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Training step", shrink=0.8)

    fig.savefig(out_dir / "distribution_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved distribution_evolution.png")


# ── Fig 2: Percentile evolution over training ──────────────────────────────────

def plot_percentile_evolution(df: pd.DataFrame, out_dir: Path):
    """10th/50th/90th percentile of each metric over steps, one line per layer."""
    layers = sorted(df.layer.unique())
    steps = sorted(df.step.unique())
    layer_colors = plt.get_cmap("tab10", len(layers))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Percentile evolution over training per layer", fontsize=13)

    for ax, (col, label) in zip(axes, METRICS.items()):
        for layer in layers:
            sub = df[df.layer == layer]
            p10 = [sub[sub.step == s][col].quantile(0.10) for s in steps]
            p50 = [sub[sub.step == s][col].quantile(0.50) for s in steps]
            p90 = [sub[sub.step == s][col].quantile(0.90) for s in steps]
            c = layer_colors(layer)
            ax.plot(steps, p50, color=c, label=f"L{layer}", linewidth=1.5)
            ax.fill_between(steps, p10, p90, color=c, alpha=0.15)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(col)

    handles = [plt.Line2D([0], [0], color=layer_colors(l), label=f"Layer {l}")
               for l in layers]
    fig.legend(handles=handles, loc="lower center", ncol=len(layers),
               bbox_to_anchor=(0.5, -0.05), fontsize=8)

    fig.savefig(out_dir / "percentile_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved percentile_evolution.png")


# ── Fig 3: Layer comparison at first and last step ────────────────────────────

def plot_layer_comparison(df: pd.DataFrame, out_dir: Path):
    """Box plots per layer for each metric, at first and last step."""
    steps = sorted(df.step.unique())
    first, last = steps[0], steps[-1]
    layers = sorted(df.layer.unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    fig.suptitle(f"Per-layer distributions at step {first} (top) and {last} (bottom)", fontsize=13)

    for col_idx, (col, label) in enumerate(METRICS.items()):
        for row_idx, step in enumerate([first, last]):
            ax = axes[row_idx, col_idx]
            data = [df[(df.step == step) & (df.layer == l)][col].values for l in layers]
            ax.boxplot(data, tick_labels=[f"L{l}" for l in layers], showfliers=False)
            ax.set_xlabel("Layer")
            ax.set_ylabel(label)
            if row_idx == 0:
                ax.set_title(col)

    fig.savefig(out_dir / "layer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved layer_comparison.png")


# ── Fig 4: Joint scatter pct_active vs mean_abs_when_active ───────────────────

def plot_joint_scatter(df: pd.DataFrame, out_dir: Path):
    """Scatter of pct_active vs mean_abs_when_active, coloured by layer."""
    steps = sorted(df.step.unique())
    first, last = steps[0], steps[-1]
    layers = sorted(df.layer.unique())
    layer_colors = plt.get_cmap("tab10", len(layers))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("pct_active vs mean_abs_when_active, coloured by layer", fontsize=13)

    for ax, step in zip(axes, [first, last]):
        sub = df[df.step == step]
        for layer in layers:
            lsub = sub[sub.layer == layer]
            ax.scatter(lsub["pct_active"], lsub["mean_abs_when_active"],
                       s=2, alpha=0.4, color=layer_colors(layer), label=f"L{layer}")
        ax.set_xlabel("pct_active")
        ax.set_ylabel("mean_abs_when_active")
        ax.set_ylim(0, 3)
        ax.set_title(f"Step {step}")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          color=layer_colors(l), label=f"Layer {l}", markersize=5)
               for l in layers]
    fig.legend(handles=handles, loc="lower center", ncol=len(layers),
               bbox_to_anchor=(0.5, -0.04), fontsize=8)

    fig.savefig(out_dir / "joint_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved joint_scatter.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=str, help="Path to mlp_stats CSV")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory to save PNGs (default: same dir as input)")
    args = p.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    df = load(input_path)
    steps = sorted(df.step.unique())
    layers = sorted(df.layer.unique())
    print(f"  {len(steps)} steps, {len(layers)} layers, "
          f"{df[df.step == steps[0]].groupby('layer').size().iloc[0]} neurons/layer")

    plot_distribution_evolution(df, out_dir)
    plot_percentile_evolution(df, out_dir)
    plot_layer_comparison(df, out_dir)
    plot_joint_scatter(df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
