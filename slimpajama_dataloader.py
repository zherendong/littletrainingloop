"""
Data loader for SlimPajama dataset turning SlimPajama dictionaries into text to train on.
"""

from typing import Any

import language_model_dataloader
from language_model_basics import LanguageModelTrainingConfig
from training_basics import DataProvider


def extract_slimpajama_data(row: dict) -> str:
    """Extract text from a row."""
    return row["text"]


def create_slimpajama_dataloader(
    config: LanguageModelTrainingConfig,
    split: str = "train",
    path: str = "data/slimpajama",
) -> DataProvider[dict[str, Any]]:
    """Load SlimPajama dataset."""
    raw_data_loader = language_model_dataloader.JSONLDataLoader(
        config, f"{path}_{split}"
    )
    tokenizer = language_model_dataloader.default_tokenizer()
    if tokenizer.n_vocab > config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.n_vocab}) is larger than the model vocab size ({config.vocab_size})"
        )
    tokenized_data_loader = language_model_dataloader.TokenizedDataLoader(
        config, raw_data_loader, tokenizer, extract_slimpajama_data
    )
    batched_data_loader = language_model_dataloader.BatchedDataLoader(
        config, tokenized_data_loader, tokenizer, split=split
    )
    return batched_data_loader
