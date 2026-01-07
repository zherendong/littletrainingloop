"""
Data loader for JSONL files.
"""

import glob
import json
from typing import Any, Callable, Iterable, TypeVar, Generic
import time

import numpy as np
import tiktoken

from training_loop import DataProvider
from language_model_basics import LMData, LanguageModelTrainingConfig
import prng
import multiprocess_iterable


def default_tokenizer():
    return tiktoken.get_encoding("cl100k_base")  # used for gpt-3.5 ad gpt-4


def get_default_tokenizer_vocab() -> list[str]:
    tok = default_tokenizer()
    vocab = []
    for token in range(tok.n_vocab):
        try:
            vocab.append(tok.decode([token]))
        except KeyError:
            print(f"Could not decode token {token}")
            vocab.append("<unk>")
    assert len(vocab) == tok.n_vocab
    return vocab


class JSONLDataLoader(DataProvider[dict[str, Any]]):
    """Loads a directory of JSONL files and returns a single iterator of dictionaries."""

    def __init__(self, config: LanguageModelTrainingConfig, path: str):
        self.path = path
        self.config = config

    def generate(self) -> Iterable[dict[str, Any]]:
        """Load data from JSONL files."""
        files = list(glob.glob(f"{self.path}/*.jsonl"))
        # creating a new prng here to make sure the shuffle is identical on repeated
        # calls to generate()
        shuffler = prng.PRNG(self.config.seed + 52135)
        shuffler.shuffle(files)

        for file_idx, file in enumerate(files):
            with open(file, "r") as f:
                for line in f:
                    yield json.loads(line)
            read_fraction = file_idx / len(files)
            print(f"Read ~{100 * read_fraction:.2f}% of the dataset")

    def get_name(self) -> str:
        return f"JSONL dataset ({self.path=})"


class MixedDataLoader(DataProvider[dict[str, Any]]):
    """Mixes multiple data loaders together."""

    def __init__(
        self,
        data_loaders: list[DataProvider[dict[str, Any]]],
        weights: list[float],
        name: str | None = None,
    ):
        self.data_loaders = data_loaders
        self.weights = weights
        self.name = name

    def generate(self) -> Iterable[dict[str, Any]]:
        """Mix data from multiple loaders deterministically according to weights.

        The algorithm maintains cumulative counts for each iterator and always
        selects the iterator that is most behind its target proportion. This
        ensures deterministic, reproducible mixing that respects the weights.

        When a dataloader is exhausted, it is automatically restarted by calling
        generate() again, allowing for infinite iteration over finite datasets.
        """
        iterators = [iter(loader.generate()) for loader in self.data_loaders]
        weights = np.array(self.weights, dtype=np.float64) / sum(self.weights)

        # Track how many items we've pulled from each iterator
        counts = np.zeros(len(iterators), dtype=np.int64)

        while True:
            # Calculate the ratio counts[i] / weights[i] for each iterator
            # We want to pull from the iterator with the smallest ratio
            # (the one most "behind" its target proportion)
            ratios = counts / weights
            selected_idx = np.argmin(ratios)

            # Try to get the next item from the selected iterator
            try:
                item = next(iterators[selected_idx])
                counts[selected_idx] += 1
                yield item
            except StopIteration:
                # This iterator is exhausted, restart it by calling generate() again
                iterators[selected_idx] = iter(
                    self.data_loaders[selected_idx].generate()
                )
                # Try again to get an item from the restarted iterator
                try:
                    item = next(iterators[selected_idx])
                    counts[selected_idx] += 1
                    yield item
                except StopIteration:
                    # The dataloader is empty (generates nothing), this is an error
                    raise ValueError(
                        f"Dataloader {self.data_loaders[selected_idx].get_name()} "
                        "is empty and cannot be used in MixedDataLoader"
                    )

    def get_name(self) -> str:
        """Name of the dataset"""
        if self.name:
            return self.name
        loader_names = ", ".join([loader.get_name() for loader in self.data_loaders])
        return f"MixedDataLoader([{loader_names}], weights={self.weights})"


U = TypeVar("U")


class TokenizedDataLoader(DataProvider[dict[str, Any]], Generic[U]):
    """Tokenizes text data on the fly and returns batches of token ids."""

    def __init__(
        self,
        config: LanguageModelTrainingConfig,
        raw_data_loader: DataProvider[U],
        tokenizer: tiktoken.Encoding,
        data_to_text: Callable[[U], str],
        data_to_input: Callable[[U], str] | None = None,
        pad_to_multiple_of: int = 1,
        append_eot: bool = False,
    ):
        """
        Args:
            config: Training configuration
            raw_data_loader: Data loader returning raw data
            tokenizer: Tokenizer
            data_to_text: Function to extract text from raw data
            data_to_input: Optional function to extract input from raw data.
                "input", as opposed to "text" is masked out.
            pad_to_multiple_of: Pad the number of tokens to a multiple of this number.
                This helps with evaluations where the same information shouldn't be
                repeated within the same batch.
            append_eot: Append an EOT token to the end of each sequence.
        """
        self.config = config
        self.raw_data_loader = raw_data_loader
        self.tokenizer = tokenizer
        self.data_to_text = data_to_text
        self.data_to_input = data_to_input
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token = tokenizer.eot_token
        self.append_eot = append_eot

    def generate(self) -> Iterable[dict[str, Any]]:
        """Create a fresh iterator."""
        for data in self.raw_data_loader.generate():
            text = self.data_to_text(data)
            tokens = self.tokenizer.encode(text, disallowed_special=())
            if self.append_eot:
                tokens.append(self.tokenizer.eot_token)
            mask = [1.0] * len(tokens)
            if self.data_to_input is not None:
                masked_input = self.data_to_input(data)
                text = masked_input + text
                masked_tokens = self.tokenizer.encode(
                    masked_input, disallowed_special=()
                )
                num_masked_tokens = len(masked_tokens)
                ### start of optimization
                # More efficien variant of the following line:
                # tokens = masked_tokens + tokens
                concatenated_tokens = masked_tokens
                concatenated_tokens.extend(tokens)
                tokens = concatenated_tokens
                ### end of optimization
                mask = [0.0] * num_masked_tokens + mask
            if self.pad_to_multiple_of > 1:
                num_pad = (
                    self.pad_to_multiple_of - len(tokens) % self.pad_to_multiple_of
                ) % self.pad_to_multiple_of
                tokens += [self.pad_token] * num_pad
                mask += [0.0] * num_pad
            text_per_token = [self.tokenizer.decode([t]) for t in tokens]
            yield {
                "tokens": tokens,
                "raw_text": text,
                "text_per_token": text_per_token,
                "mask": mask,
            }

    def get_name(self) -> str:
        return f"Tokenized dataset ({self.raw_data_loader.get_name()})"


class BatchedDataLoader(DataProvider[LMData]):
    """Data loader creating batches.

    Tokenizes text data on the fly and returns batches of token ids.

    If an example is too long, it is spread out over multiple batches,
    but always stays in the same batch index.
    """

    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        tokenized_data_loader: TokenizedDataLoader,
        tokenizer: tiktoken.Encoding,
        split: str = "train",
        name: str | None = None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenized_data_loader = tokenized_data_loader
        self.tokenizer = tokenizer
        self.split = split
        self.name = name

    def generate(self) -> Iterable[LMData]:
        """Create a fresh iterator."""
        global_data_stream = iter(self.tokenized_data_loader.generate())
        rest_data_per_batch = [None] * self.batch_size

        while True:
            shape = (self.batch_size, self.sequence_length)
            inputs = np.zeros(shape, dtype=np.int32)
            targets = np.zeros(shape, dtype=np.int32)
            loss_mask = np.ones(shape, dtype=np.float32)
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
                mask = []
                text_per_tokens = []

                for data in continued_data_stream(rest_data):
                    free_space_in_batch_item = self.sequence_length - len(tokens)
                    tokens.extend(data["tokens"][:free_space_in_batch_item])
                    mask.extend(data["mask"][:free_space_in_batch_item])
                    text_per_tokens.extend(
                        data["text_per_token"][:free_space_in_batch_item]
                    )

                    rest_data = None
                    if len(data["tokens"]) > free_space_in_batch_item:
                        rest_data = {
                            "tokens": data["tokens"][free_space_in_batch_item:],
                            "mask": data["mask"][free_space_in_batch_item:],
                            "text_per_token": data["text_per_token"][
                                free_space_in_batch_item:
                            ],
                        }
                    rest_data_per_batch[batch_idx] = rest_data

                    if len(tokens) == self.sequence_length:
                        break

                assert (
                    len(tokens) == self.sequence_length
                ), f"got {len(tokens)}; expected {self.sequence_length} tokens."
                assert len(text_per_tokens) == self.sequence_length
                inputs[batch_idx] = tokens
                # targets are the tokens shifted by one
                targets[batch_idx] = tokens[1:] + [self.tokenizer.eot_token]
                # shift the mask by one to lign up with targets, append a zero to also mask out the EOT token
                loss_mask[batch_idx] = mask[1:] + [0.0]
                metadata["text_per_tokens"].append(text_per_tokens)

            yield LMData(
                inputs,
                targets,
                loss_mask,
                metadata=metadata,
            )

    def get_name(self) -> str:
        """Name of the dataset"""
        if self.name:
            return self.name
        return f"Batched LM dataset ({self.batch_size=}, {self.tokenized_data_loader.get_name()})"


T = TypeVar("T")


class MultiProcessDataloader(DataProvider[T]):
    """Wrap a dataloader to run it in a separate process."""

    def __init__(
        self,
        dataloader_factory: Callable[..., Iterable[T]],
        kwargs: dict[str, Any],
        prefetch: int,
        name: str | None = None,
    ):
        self.dataloader_factory = dataloader_factory
        self.kwargs = kwargs
        self.prefetch = prefetch
        self.name = name

    def generate(self) -> Iterable[T]:
        """Create a fresh iterator."""
        yield from multiprocess_iterable.GeneratorProcess(
            self.dataloader_factory, self.kwargs, prefetch=self.prefetch
        )

    def get_name(self) -> str:
        """Name of the dataset"""
        if self.name:
            return self.name
        return f"MultiProcessDataloader({self.dataloader_factory.__name__})"
