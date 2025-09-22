"""
Data loader for JSONL files.
"""

import glob
import json
import logging
from typing import Any, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
import torch
import tiktoken

from training_loop import DataProvider
from language_model_training import DataItem, LanguageModelTrainingConfig

import prng


def default_tokenizer():
    return tiktoken.get_encoding("cl100k_base")  # used for gpt-3.5 ad gpt-4


class JSONLDataLoader(DataProvider[dict[str, Any]]):
    """Loads a directory of JSONL files and returns a single iterator of dictionaries."""

    def __init__(self, config: LanguageModelTrainingConfig, path: str):
        self.path = path
        self.config = config
        self.prng = prng.PRNG(config.seed + 52135)

    def generate(self) -> Iterable[dict[str, Any]]:
        """Load data from JSONL files."""
        files = list(glob.glob(f"{self.path}/*.jsonl"))
        self.prng.shuffle(files)

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    yield json.loads(line)

    def get_name(self) -> str:
        return f"JSONL dataset ({self.path=})"


class TokenizedDataLoader(DataProvider[dict[str, Any]]):
    """Tokenizes text data on the fly and returns batches of token ids."""

    def __init__(
        self,
        config: LanguageModelTrainingConfig,
        raw_data_loader: DataProvider[dict[str, Any]],
        tokenizer: tiktoken.Encoding,
        data_to_text: Callable[[dict[str, Any]], str],
    ):
        self.config = config
        self.raw_data_loader = raw_data_loader
        self.tokenizer = tokenizer
        self.data_to_text = data_to_text

    def generate(self) -> Iterable[dict[str, Any]]:
        """Create a fresh iterator."""
        for data in self.raw_data_loader.generate():
            text = self.data_to_text(data)
            tokens = self.tokenizer.encode(text)
            text_per_token = [self.tokenizer.decode([t]) for t in tokens]
            yield {
                "tokens": tokens,
                "raw_text": text,
                "text_per_token": text_per_token,
            }

    def get_name(self) -> str:
        return f"Tokenized dataset ({self.raw_data_loader.get_name()})"


class BatchedDataLoader(DataProvider[DataItem]):
    """Data loader creating batches.

    Tokenizes text data on the fly and returns batches of token ids.

    If an example is too long, it is spread out over multiple batches,
    but always stays in the same batch index.
    """

    def __init__(
        self,
        config: LanguageModelTrainingConfig,
        tokenized_data_loader: DataProvider[dict[str, Any]],
        tokenizer: tiktoken.Encoding,
        split: str = "train",
    ):
        self.config = config

        self.tokenized_data_loader = tokenized_data_loader
        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = (
            config.batch_size
            if split == "train"
            else config.training_config.eval_batch_size
        )

    def generate(self) -> Iterable[DataItem]:
        """Create a fresh iterator."""
        global_data_stream = iter(self.tokenized_data_loader.generate())
        rest_data_per_batch = [None] * self.batch_size

        while True:
            shape = (self.batch_size, self.config.sequence_length)
            inputs = torch.zeros(shape, dtype=torch.int32)
            targets = torch.zeros(shape, dtype=torch.int32)
            loss_mask = torch.ones(shape, dtype=torch.float32)
            metadata = {"text_per_tokens": []}

            def continued_data_stream(rest_data) -> Iterable[dict[str, Any]]:
                """Stream data, starting with rest_data if any."""
                if rest_data is not None:
                    yield rest_data  # continue processing tokens in rest_data
                while True:
                    try:
                        yield next(global_data_stream)
                    except StopIteration:
                        break

            for batch_idx, rest_data in enumerate(rest_data_per_batch):
                tokens = []
                text_per_tokens = []

                for data in continued_data_stream(rest_data):
                    free_space_in_batch_item = self.config.sequence_length - len(tokens)
                    tokens.extend(data["tokens"][:free_space_in_batch_item])
                    text_per_tokens.extend(
                        data["text_per_token"][:free_space_in_batch_item]
                    )

                    rest_data = None
                    if len(data["tokens"]) > free_space_in_batch_item:
                        rest_data = {
                            "tokens": data["tokens"][free_space_in_batch_item:],
                            "text_per_token": data["text_per_token"][
                                free_space_in_batch_item:
                            ],
                        }
                    rest_data_per_batch[batch_idx] = rest_data

                    if len(tokens) == self.config.sequence_length:
                        break

                assert len(tokens) == self.config.sequence_length, (
                    f"got {len(tokens)}; expected {self.config.sequence_length} tokens."
                )
                assert len(text_per_tokens) == self.config.sequence_length
                inputs[batch_idx] = torch.tensor(tokens)
                target_tokens = tokens[1:] + [self.tokenizer.eot_token]
                targets[batch_idx] = torch.tensor(target_tokens)
                loss_mask[batch_idx, -1] = 0.0  # mask out the EOT token
                metadata["text_per_tokens"].append(text_per_tokens)

            yield DataItem(
                inputs,
                targets,
                loss_mask,
                metadata=metadata,
            )

    def get_name(self) -> str:
        """Name of the dataset"""
        return f"Batched LM dataset ({self.config.batch_size=}, {self.tokenized_data_loader.get_name()})"
