"""Fit shifted power law curves to scaling law data and generate plots.

This script reads prepared scaling law data (pflops vs final_loss),
fits shifted power law models, and generates plots. Supports multiple
datasets for comparison.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


def shifted_power_law(x, a, b, c):
    """Shifted power law: y = a * x^b + c"""
    return a * x**b + c


def fit_single_dataset(
    data_file: str, fixed_c: float | None = None, min_pflops: float = 0
):
    """Fit shifted power law to a single dataset.

    Args:
        data_file: Path to CSV file with pflops and final_loss columns
        fixed_c: If provided, fix the c parameter to this value and only fit a and b
        min_pflops: Minimum PFLOPs threshold - datapoints below this are excluded

    Returns:
        Dictionary with data, fit parameters, and metadata
    """
    # Load data (skip lines starting with #)
    print(f"\nLoading data from: {data_file}")
    data = pd.read_csv(data_file, comment="#")

    if "pflops" not in data.columns or "final_loss" not in data.columns:
        raise ValueError(
            f"Data file must contain 'pflops' and 'final_loss' columns. "
            f"Found columns: {data.columns.tolist()}"
        )

    x = data["pflops"].values
    y = data["final_loss"].values

    print(f"  Loaded {len(x)} data points")
    print(f"  PFLOPs range: {x.min():.2f} - {x.max():.2f}")

    # Filter by minimum PFLOPs
    if min_pflops > 0:
        mask = x >= min_pflops
        x = x[mask]
        y = y[mask]
        print(f"  After filtering (min_pflops={min_pflops}): {len(x)} data points")
        if len(x) == 0:
            raise ValueError(
                f"No data points remain after filtering with min_pflops={min_pflops}"
            )
        print(f"  Filtered PFLOPs range: {x.min():.2f} - {x.max():.2f}")

    print(f"  Loss range: {y.min():.4f} - {y.max():.4f}")

    # Fit shifted power law
    if fixed_c is None:
        print("  Fitting shifted power law (all parameters)...")
        try:
            # Use median loss as initial guess for c
            c_init = np.median(y)
            # Set bounds: c must be non-negative
            # bounds format: ([lower bounds], [upper bounds])
            # a: no lower bound, b: no bounds, c: >= 0
            popt_shifted, pcov_shifted = curve_fit(
                shifted_power_law,
                x,
                y,
                p0=[1, -0.1, c_init],
                bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
                maxfev=10000,
            )
            print(
                f"    Parameters: a={popt_shifted[0]:.6f}, "
                f"b={popt_shifted[1]:.6f}, c={popt_shifted[2]:.6f}"
            )

            return {
                "file": data_file,
                "label": Path(data_file).stem,
                "x": x,
                "y": y,
                "params": popt_shifted,
                "success": True,
                "c_fixed": False,
            }
        except Exception as e:
            print(f"    Warning: Fit failed: {e}")
            return {
                "file": data_file,
                "label": Path(data_file).stem,
                "x": x,
                "y": y,
                "params": None,
                "success": False,
                "c_fixed": False,
            }
    else:
        print(f"  Fitting shifted power law (c fixed to {fixed_c:.6f})...")
        try:
            # Fit only a and b with c fixed
            def power_law_fixed_c(x_val, a, b):
                return a * x_val**b + fixed_c

            popt_ab, pcov_ab = curve_fit(
                power_law_fixed_c, x, y, p0=[1, -0.1], maxfev=10000
            )
            # Combine with fixed c to get full parameter set
            popt_shifted = np.array([popt_ab[0], popt_ab[1], fixed_c])
            print(
                f"    Parameters: a={popt_shifted[0]:.6f}, "
                f"b={popt_shifted[1]:.6f}, c={popt_shifted[2]:.6f} (fixed)"
            )

            return {
                "file": data_file,
                "label": Path(data_file).stem,
                "x": x,
                "y": y,
                "params": popt_shifted,
                "success": True,
                "c_fixed": True,
            }
        except Exception as e:
            print(f"    Warning: Fit failed: {e}")
            return {
                "file": data_file,
                "label": Path(data_file).stem,
                "x": x,
                "y": y,
                "params": None,
                "success": False,
                "c_fixed": True,
            }


def fit_scaling_laws(data_files: list[str], output_base: str, min_pflops: float = 0):
    """Fit shifted power laws to multiple datasets and generate comparison plots.

    For multiple datasets, the c parameter from the first dataset is used for all
    subsequent datasets, since c represents the irreducible loss floor which is
    a property of the dataset/task, not the model.

    Args:
        data_files: List of paths to CSV files with pflops and final_loss columns
        output_base: Base name for output files (plots and fit parameters)
        min_pflops: Minimum PFLOPs threshold - datapoints below this are excluded
    """
    print(f"Fitting scaling laws for {len(data_files)} dataset(s)...")
    if min_pflops > 0:
        print(f"Filtering datapoints with PFLOPs < {min_pflops}")

    # Fit each dataset
    results = []
    shared_c = None

    for i, data_file in enumerate(data_files):
        if i == 0:
            # First dataset: fit all parameters
            result = fit_single_dataset(data_file, fixed_c=None, min_pflops=min_pflops)
            if result["success"]:
                shared_c = result["params"][2]
                print(
                    f"\n  Using c={shared_c:.6f} from first dataset for all subsequent fits"
                )
        else:
            # Subsequent datasets: use c from first dataset
            result = fit_single_dataset(
                data_file, fixed_c=shared_c, min_pflops=min_pflops
            )

        results.append(result)

    # Determine global x-axis range for plotting
    all_x = np.concatenate([r["x"] for r in results])
    x_min, x_max = all_x.min(), all_x.max()
    x_pred = np.logspace(np.log10(x_min), np.log10(x_max * 10), 200)

    # Define colors for different datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot each dataset
    for i, result in enumerate(results):
        color = colors[i]
        label = result["label"]

        # Plot data points
        plt.scatter(
            result["x"],
            result["y"],
            color=color,
            s=100,
            label=f"{label} (data)",
            zorder=5,
            alpha=0.7,
        )

        # Plot fitted curve if successful
        if result["success"]:
            params = result["params"]
            y_fit = shifted_power_law(x_pred, *params)
            plt.plot(
                x_pred,
                y_fit,
                "-",
                color=color,
                linewidth=2,
                label=(
                    f"{label} fit: a={params[0]:.3f}, "
                    f"b={params[1]:.3f}, c={params[2]:.3f}"
                ),
                alpha=0.8,
            )

    plt.xscale("log")
    plt.xlabel("Compute (PFLOPs)", fontsize=12)
    plt.ylabel("Final Evaluation Loss", fontsize=12)

    # Add note about shared c in title if applicable
    if len(results) > 1 and shared_c is not None:
        title = f"Scaling Law Comparison (Shifted Power Law, shared c={shared_c:.3f})"
    else:
        title = "Scaling Law Comparison (Shifted Power Law)"
    plt.title(title, fontsize=14)

    plt.legend(fontsize=9, loc="best")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    # Save plot
    plot_path = f"{output_base}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    # Save fit parameters to text file
    params_path = f"{output_base}_params.txt"
    with open(params_path, "w") as f:
        f.write("Scaling Law Fit Parameters (Shifted Power Law)\n")
        f.write("=" * 70 + "\n\n")

        # Note about shared c parameter
        if len(results) > 1 and shared_c is not None:
            f.write(
                f"NOTE: All datasets use shared c = {shared_c:.6f} from first dataset\n"
            )
            f.write("      (c represents the irreducible loss floor)\n\n")

        for i, result in enumerate(results):
            f.write(f"Dataset {i+1}: {result['file']}\n")
            f.write(f"  Label: {result['label']}\n")
            f.write(f"  Number of data points: {len(result['x'])}\n")
            f.write(
                f"  PFLOPs range: {result['x'].min():.2f} - {result['x'].max():.2f}\n"
            )
            f.write(
                f"  Loss range: {result['y'].min():.4f} - {result['y'].max():.4f}\n"
            )

            if result["success"]:
                params = result["params"]
                c_note = " (shared)" if result.get("c_fixed", False) else ""
                f.write(f"  Shifted Power Law (y = a * x^b + c):\n")
                f.write(f"    a = {params[0]:.6f}\n")
                f.write(f"    b = {params[1]:.6f}\n")
                f.write(f"    c = {params[2]:.6f}{c_note}\n")
            else:
                f.write("  Fit: FAILED\n")
            f.write("\n")

    print(f"Fit parameters saved to: {params_path}")

    # Save fit parameters to CSV
    params_csv_path = f"{output_base}_params.csv"
    params_data = []

    for result in results:
        if result["success"]:
            params = result["params"]
            params_data.append(
                {
                    "dataset": result["label"],
                    "file": result["file"],
                    "a": params[0],
                    "b": params[1],
                    "c": params[2],
                    "c_fixed": result.get("c_fixed", False),
                    "n_points": len(result["x"]),
                    "pflops_min": result["x"].min(),
                    "pflops_max": result["x"].max(),
                    "loss_min": result["y"].min(),
                    "loss_max": result["y"].max(),
                }
            )

    if params_data:
        params_df = pd.DataFrame(params_data)
        params_df.to_csv(params_csv_path, index=False)
        print(f"Fit parameters CSV saved to: {params_csv_path}")


def main(data_files: list[str], output_base: str | None = None, min_pflops: float = 0):
    """Main function to fit scaling laws and generate plots.

    Args:
        data_files: List of paths to CSV files with prepared scaling data
        output_base: Base name for output files (default: derived from input files)
        min_pflops: Minimum PFLOPs threshold - datapoints below this are excluded (default: 0)
    """
    # Set default output base if not specified
    if output_base is None:
        if len(data_files) == 1:
            # Single file: use filename as base
            data_file = data_files[0]
            if data_file.endswith(".csv"):
                output_base = data_file[:-4] + "_fit"
            else:
                output_base = data_file + "_fit"
        else:
            # Multiple files: use generic name
            output_base = "scaling_analysis/scaling_law_comparison"

    fit_scaling_laws(data_files, output_base, min_pflops=min_pflops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit shifted power law curves to scaling law data and generate comparison plots"
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to CSV file(s) with prepared scaling data (pflops, final_loss columns). "
        "Multiple files can be specified for comparison.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base name for output files (default: <data_file>_fit for single file, "
        "scaling_analysis/scaling_law_comparison for multiple files)",
    )
    parser.add_argument(
        "--min-pflops",
        type=float,
        default=500,
        help="Minimum PFLOPs threshold - datapoints below this are excluded (default: 500)",
    )

    args = parser.parse_args()

    main(data_files=args.data, output_base=args.output, min_pflops=args.min_pflops)
