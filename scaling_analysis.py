"""Entry point for extracting data from neptune on scaling experiments."""

import argparse
import os
import pandas as pd
import neptune
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Sequence


def retrieve_run_ids(neptune_api_token: str):
    """Retrieve all run ids in the project"""
    neptune_project = neptune.init_project(
        project="markusrabeworkspace/training-exploration",
        api_token=neptune_api_token,
    )
    table = neptune_project.fetch_runs_table()
    table = table.to_pandas()
    return list(table["sys/id"].values)


def extract_data(neptune_api_token: str, run_id: str) -> pd.DataFrame:
    """Extract data from a single run"""
    neptune_run = neptune.init_run(
        with_id=run_id,
        project="markusrabeworkspace/training-exploration",
        api_token=neptune_api_token,
    )

    out = neptune_run["train/loss"].fetch_values(include_timestamp=False)
    # also get num_tokens and training_flops from the run
    print(out)
    return out


def fetch_loss_vs_flops(token, run_id):
    """Extract loss vs flops from a single run"""
    print(f"Fetching data for run {run_id}")
    run = neptune.init_run(
        with_id=run_id,
        project="markusrabeworkspace/training-exploration",
        api_token=token,
    )
    loss = (
        # run["train/loss"]
        run["eval/SlimPajama_validation/loss"]
        .fetch_values(include_timestamp=False)
        .rename(columns={"value": "eval/SlimPajama_validation/loss"})
    )
    flops = (
        run["train/pflops_total"]
        .fetch_values(include_timestamp=False)
        .rename(columns={"value": "train/pflops_total"})
    )
    df = pd.merge(
        loss[["step", "eval/SlimPajama_validation/loss"]],
        flops[["step", "train/pflops_total"]],
        on="step",
        how="inner",
    )
    return df


def main(output_base: str, run_ids: Sequence[str]):
    """Main function"""
    load_dotenv(dotenv_path=os.path.expanduser("~/.neptune/.env"))
    neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]

    if not run_ids:
        run_ids = retrieve_run_ids(neptune_api_token)
    for run_id in run_ids:
        df = fetch_loss_vs_flops(neptune_api_token, run_id)
        df.to_csv(f"scaling_data/{output_base}_{run_id}.csv", index=False)
        plt.semilogx(
            df["train/pflops_total"],
            df["eval/SlimPajama_validation/loss"],
            label=run_id,
        )

    plt.xlabel("PFLOPs total")
    plt.ylabel("Validation loss")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()

    # ... after plt.tight_layout()
    plot_path = f"scaling_data/{output_base}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ids", type=str, nargs="+")
    parser.add_argument("--output_base", type=str)
    args = parser.parse_args()
    main(output_base=args.output_base, run_ids=args.run_ids)
