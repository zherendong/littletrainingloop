"""
Compute per-step activation-rate distributions from mlp_neuron_stats.csv.

For every step divisible by 100, bins all neurons (averaged across layers)
by their pct_active value into decile buckets (0–10 %, 10–20 %, … 90–100 %)
and reports the fraction of neurons falling in each bucket.

Usage
-----
python activation_distribution.py                          # defaults
python activation_distribution.py --input results/mlp_neuron_stats.csv
python activation_distribution.py --output results/act_dist.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


BINS = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
BIN_LABELS = [f"{int(BINS[i]*100)}-{int(BINS[i+1]*100)}%" for i in range(len(BINS) - 1)]


def compute_distribution(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["step"] % 100 == 0]
    rows = []
    for step, group in df.groupby("step", sort=True):
        counts, _ = np.histogram(group["pct_active"].values, bins=BINS)
        fractions = counts / counts.sum()
        row = {"step": step}
        for label, frac in zip(BIN_LABELS, fractions):
            row[label] = round(float(frac), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=str, default="./results/mlp_neuron_stats.csv")
    p.add_argument("--output", type=str, default=None, help="Optional path to write output CSV")
    args = p.parse_args()

    df = pd.read_csv(args.input)

    result = compute_distribution(df)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"Written to {out_path}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
