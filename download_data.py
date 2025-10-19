"""
Data loading and processing as a python file, not a notebook.
"""

import argparse
import json
import os
import time
import boto3
import pandas as pd
from smart_open import open
from datasets import load_dataset
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from prng import PRNG


def download_file_content(s3, file_info):
    """Download content for a single file"""
    s3_url = f"s3://softwareheritage/content/{file_info['blob_id']}"
    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        file_info["content"] = fin.read().decode(file_info["src_encoding"])
    return file_info


def download_contents(s3, executor: ThreadPoolExecutor, files: list[dict]):
    futures = as_completed(
        [executor.submit(download_file_content, s3, file) for file in files]
    )
    return futures


def split_in_batches(dataset: Iterable[dict], batch_size: int = 1000) -> Iterable[list]:
    """Split a dataset into batches"""
    dataset_iterator = iter(dataset)
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(dataset_iterator))
            except StopIteration:
                print(f"End of dataset after {len(batch)} rows")
                # raise ValueError("Unexpected end of dataset")
                return
        yield batch


def hydrate_batch(batch: list[dict], executor: ThreadPoolExecutor, s3) -> list[dict]:
    for row in batch:
        row["file_futures"] = download_contents(s3, executor, row["files"])
    for row in batch:
        row["files"] = [future.result() for future in row["file_futures"]]
        del row["file_futures"]
    return batch


def save_batch(batch: list[dict], directory: str, idx: int):
    """Save batches of data to disk"""
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/blob_{idx:05d}.jsonl", "w") as fout:
        stringified_rows = []
        for row in batch:
            for key, val in row.items():
                # if not isinstance(
                #     val, (str, int, float, list, dict, None, pd.Timestamp)
                # ):
                #     print(f"Unexpected type {type(val)} for key {key}")
                if isinstance(
                    val, pd.Timestamp
                ):  # pandas timestamps cannot be serialized
                    row[key] = val.value
            stringified_rows.append(json.dumps(row))
        full_content = "\n".join(stringified_rows)
        fout.write(full_content)
    print(f"Saved batch {idx}")


def download_slimpajama(split: str = "train", output_dir: str = "data/slimpajama"):
    # https://huggingface.co/datasets/cerebras/SlimPajama-627B
    ds = load_dataset("cerebras/SlimPajama-627B", streaming=True)
    ds_train, ds_test = ds["train"], ds["test"]

    ds = ds_train if split == "train" else ds_test

    start_time = time.time()
    batched_ds = split_in_batches(ds, batch_size=5000)
    for idx, batch in enumerate(batched_ds):
        print(f"Batch {idx}")
        path = f"{output_dir}_{split}"
        os.makedirs(path, exist_ok=True)
        save_batch(batch, path, idx)
        if idx >= 999:
            break

    total_time = time.time() - start_time
    print(f"Total time: {total_time}")

    return ds_train, ds_test


def upsample_long_popular_repos(ds: Iterable[dict]) -> Iterable[dict]:
    """Increase the relative weight of long repositories and forked repositories."""
    # If a repository has more than 100 files, include it.
    # Below 100 files, include it with probability num_files / 100.
    # A fork counts as 10 files.
    prng = PRNG(12342323)
    for row in ds:
        num_files = len(row["files"])
        score = num_files
        if row["fork_events_count"] > 0:
            score += row["fork_events_count"] * 10

        if prng.random() < score / 100:
            print(
                f"dataset with {num_files} files and {row['fork_events_count']} forks. Accepted."
            )
            yield row
        else:
            print(
                f"dataset with {num_files} files and {row['fork_events_count']} forks. Rejected."
            )


def download_stackv2(output_dir: str = "data/stackv2_long"):
    # https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids

    # load_dotenv(dotenv_path=os.path.expanduser("~/.aws/.env"))

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.client("s3")

    ds = load_dataset(
        "bigcode/the-stack-v2-train-full-ids",
        split="train",  # only train is available
        streaming=True,
    )
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=64) as executor:
        long_repos_ds = upsample_long_popular_repos(ds)
        batched_ds = split_in_batches(long_repos_ds, batch_size=100)
        for idx, batch in enumerate(batched_ds):
            batch = hydrate_batch(batch, executor, s3)
            executor.submit(save_batch, batch, directory=output_dir, idx=idx)

            if idx >= 999:
                break

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    s3.close()
    print("Done!")


def run():
    # command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slimpajama")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for the downloaded dataset (default: 'data')")
    args = parser.parse_args()

    if args.dataset == "slimpajama":
        output_dir = f"{args.output_dir}/slimpajama"
        download_slimpajama(args.split, output_dir)
    elif args.dataset == "stackv2":
        output_dir = f"{args.output_dir}/stackv2_long"
        download_stackv2(output_dir)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")


if __name__ == "__main__":
    run()
