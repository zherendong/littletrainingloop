"""Download and prepare scaling law data from Neptune experiments.

This script extracts the final evaluation loss for each experiment with a specified tag,
aggregates them, and saves the data for scaling law analysis.
"""

import argparse
import os
import pandas as pd
import neptune
from dotenv import load_dotenv
from typing import Sequence


def fetch_run_ids_by_tag(neptune_api_token: str, tag: str) -> list[str]:
    """Retrieve all run ids in the project that have the specified tag.

    Args:
        neptune_api_token: Neptune API token for authentication
        tag: Tag to filter runs by

    Returns:
        List of run IDs that have the specified tag
    """
    neptune_project = neptune.init_project(
        project="markusrabeworkspace/training-exploration",
        api_token=neptune_api_token,
    )

    # Use NQL query to filter by tag
    query = f'(`sys/tags`:stringSet CONTAINS "{tag}")'
    table = neptune_project.fetch_runs_table(query=query)
    table_df = table.to_pandas()

    print(f"Found {len(table_df)} runs with tag '{tag}'")
    return list(table_df["sys/id"].values)


def extract_final_loss_and_flops(token: str, run_id: str) -> dict | None:
    """Extract the final evaluation loss and corresponding PFLOPs from a single run.

    Args:
        token: Neptune API token
        run_id: Neptune run ID

    Returns:
        Dictionary with run_id, pflops, and final_loss, or None if data is missing
    """
    print(f"Fetching data for run {run_id}")
    try:
        run = neptune.init_run(
            with_id=run_id,
            project="markusrabeworkspace/training-exploration",
            api_token=token,
            mode="read-only",
        )

        # Fetch the evaluation loss series
        loss_df = (
            run["eval/SlimPajama_validation/loss"]
            .fetch_values(include_timestamp=False)
            .rename(columns={"value": "eval/SlimPajama_validation/loss"})
        )

        # Fetch the PFLOPs series
        flops_df = (
            run["train/pflops_total"]
            .fetch_values(include_timestamp=False)
            .rename(columns={"value": "train/pflops_total"})
        )

        # Merge on step to ensure we get matching data points
        df = pd.merge(
            loss_df[["step", "eval/SlimPajama_validation/loss"]],
            flops_df[["step", "train/pflops_total"]],
            on="step",
            how="inner",
        )

        if len(df) == 0:
            print(f"  Warning: No data found for run {run_id}")
            run.stop()
            return None

        # Get the last entry (final loss and corresponding PFLOPs)
        final_row = df.iloc[-1]
        final_loss = final_row["eval/SlimPajama_validation/loss"]
        final_pflops = final_row["train/pflops_total"]

        print(f"  Final loss: {final_loss:.4f}, PFLOPs: {final_pflops:.2f}")

        run.stop()

        return {
            "run_id": run_id,
            "pflops": final_pflops,
            "final_loss": final_loss,
        }

    except Exception as e:
        print(f"  Error fetching data for run {run_id}: {e}")
        return None


def main(tag: str, output_file: str, run_ids: Sequence[str] | None = None):
    """Main function to download and prepare scaling law data.

    Args:
        tag: Neptune tag to filter experiments by
        output_file: Path to save the prepared CSV file
        run_ids: Optional list of specific run IDs to process (overrides tag filtering)
    """
    load_dotenv(dotenv_path=os.path.expanduser("~/.neptune/.env"))
    neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]

    # Get run IDs either from explicit list or by tag
    if run_ids:
        print(f"Processing {len(run_ids)} specified runs")
        run_ids_to_process = run_ids
    else:
        print(f"Fetching runs with tag: {tag}")
        run_ids_to_process = fetch_run_ids_by_tag(neptune_api_token, tag)

    if not run_ids_to_process:
        print("No runs found to process!")
        return

    # Extract final loss and PFLOPs for each run
    results = []
    for run_id in run_ids_to_process:
        result = extract_final_loss_and_flops(neptune_api_token, run_id)
        if result is not None:
            results.append(result)

    if not results:
        print("No valid data extracted from any runs!")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values("pflops")  # Sort by PFLOPs for easier visualization

    print(f"\nExtracted data from {len(df)} runs:")
    print(df)

    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare scaling law data from Neptune experiments"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Neptune tag to filter experiments by",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: scaling_analysis/scaling_data_<tag>.csv)",
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional: specific run IDs to process (overrides tag filtering)",
    )

    args = parser.parse_args()

    # Set default output file if not specified
    if args.output is None:
        output_file = f"scaling_analysis/scaling_data_{args.tag}.csv"
    else:
        output_file = args.output

    main(tag=args.tag, output_file=output_file, run_ids=args.run_ids)
