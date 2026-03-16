"""
Plot a histogram of mean_abs activation per layer at the last logged step.

One subplot per layer, all on a single figure.

Usage
-----
python plot_neuron_stats.py
python plot_neuron_stats.py --input results/mlp_neuron_stats.csv
python plot_neuron_stats.py --output results/mean_abs_hist.png
python plot_neuron_stats.py --bins 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=str, default="./results/mlp_neuron_stats.csv")
    p.add_argument("--output", type=str, default=None, help="Save figure to this path instead of showing it")
    p.add_argument("--bins",   type=int, default=40,   help="Number of histogram bins (default: 40)")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    last_step = df["step"].max()
    df = df[df["step"] == last_step]

    layers = sorted(df["layer"].unique())
    n_layers = len(layers)

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        values = df[df["layer"] == layer]["mean_abs"]
        ax.hist(values, bins=args.bins, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("mean |activation|")
        if layer == layers[0]:
            ax.set_ylabel("neuron count")

    fig.suptitle(f"Neuron mean |activation| distribution  (step {last_step})", y=1.02)
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
