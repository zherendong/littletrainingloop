"""Entry point for extracting data from neptune on scaling experiments."""

import argparse
import os
import pandas as pd
import neptune
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


def main(output_file: str, run_ids: Sequence[str]):
    """Main function"""
    load_dotenv(dotenv_path=os.path.expanduser("~/.neptune/.env"))
    neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]

    if not run_ids:
        run_ids = retrieve_run_ids(neptune_api_token)
    for run_id in run_ids:
        df = extract_data(neptune_api_token, run_id)
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ids", type=str, nargs="+")
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    main(output_file=args.output_file, run_ids=args.run_ids)
