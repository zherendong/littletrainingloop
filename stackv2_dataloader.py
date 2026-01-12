"""
Data loader for Stack v2 dataset turning stackv2 dictionaries into text to train on.


Expected labels:

repo_name (string): Repository name on GitHub.
repo_url (string): URL of the repository on GitHub.
snapshot_id (string): SWH snapshot ID.
revision_id (string): SWH revision (commit) ID.
directory_id (string): SWH ID of the root directory of the repository.
branch_name (string): Repository branch name.
visit_date (timestamp[ns]): SWH crawl (snapshot) timestamp.
revision_date (timestamp[ns]): SWH revision (commit) timestamp.
committer_date (timestamp[ns]): SWH revision (commit) timestamp reported by the committer.
github_id (int64): GitHub identifier for the repository.
star_events_count (int64): Number of stars calculated from GHArchive events.
fork_events_count (int64): Number of forks calculated from GHArchive events.
gha_license_id (string): GHArchive SPDX license identifier, None if the repo is missing.
gha_created_at (timestamp[ns]): Timestamp of repository creation on GitHub, None if the repo is missing.
gha_updated_at (timestamp[ns]): Timestamp of the latest update on GitHub, None if the repo is missing.
gha_pushed_at (timestamp[ns]): Timestamp of the last push on GitHub, None if the repo is missing.
gha_language (string): Repository's primary programming language on GitHub, None if the repo is missing.
files (list): List of files in the repository.
    blob_id (string): Software Heritage (SWH) ID of the file on AWS S3.
    path (string): The file path within the repository.
    content_id (string): SWH content ID.
    language (string): Programming language of the file, detected by go-enry / linguist.
    length_bytes (int64): Length of the file content in UTF-8 bytes.
    detected_licenses (string[]): List of licenses (SPDX) detected by ScanCode.
    license_type (string): Inferred license type (permissive or no_license).
    src_encoding (string): Original encoding of the file content before converting to UTF-8.
    is_vendor (bool): Indicator of vendor file (external library), detected by go-enry.
    is_generated (bool): Indicator of generated file (external library), detected by go-enry.
    alphanum_fraction (float32): Fraction of alphanumeric characters in the file content.
    alpha_fraction (float32): Fraction of alphabetic characters in the file content.
    num_lines (int32): Number of lines in the file.
    avg_line_length (float32): Average length of lines in the file.
    max_line_length (int32): Maximum length of a line in the file.
num_files (int64): Number of files in the repository.
"""

import language_model_dataloader
from language_model_basics import LanguageModelTrainingConfig


def preamble(row: dict) -> str:
    """Generate preamble for a row."""
    return f"=== Repository {row['repo_name']}, branch {row['branch_name']} with {row['num_files']} files. ==="


def extract_stackv2(row: dict) -> str:
    """Extract text from a row."""

    s = [preamble(row)]
    for file in row["files"]:
        s.append(f"=== File {file['path']} ({file['language']}) ===")
        s.append(file["content"])
        s.append("\n")
    return "\n".join(s)


def create_stackv2_dataloader(
    config: LanguageModelTrainingConfig,
    path: str = "data/stackv2_long",
    split: str = "train",
) -> language_model_dataloader.BatchedDataLoader:
    """Load Stack v2 dataset."""
    if split != "train":
        print(
            f"Stackv2: Split {split} requested. Only train split is available. Using train instead."
        )
    raw_data_loader = language_model_dataloader.JSONLDataLoader(config, path)
    tokenizer = language_model_dataloader.default_tokenizer()
    if tokenizer.n_vocab > config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.n_vocab}) is larger than the model vocab size ({config.vocab_size})"
        )
    tokenized_data_loader = language_model_dataloader.TokenizedDataLoader(
        config,
        raw_data_loader,
        tokenizer,
        extract_stackv2,
        append_eot=config.separate_data_with_eot,
    )
    batched_data_loader = language_model_dataloader.BatchedDataLoader(
        config.batch_size if split == "train" else config.eval_config.batch_size,
        (
            config.sequence_length
            if split == "train"
            else config.eval_config.sequence_length
        ),
        tokenized_data_loader,
        tokenizer,
        name="Stackv2",
    )
    return batched_data_loader
