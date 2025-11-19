"""
Test script for the language model dataloader.
"""

from typing import Any, Iterable

from training_basics import DataProvider, TrainingConfig
from language_model_basics import EvalConfig
from language_model_dataloader import (
    BatchedDataLoader,
    TokenizedDataLoader,
    default_tokenizer,
)
from language_model_basics import LMData, LanguageModelTrainingConfig
import torch


class DummyRawDataProvider(DataProvider[dict[str, Any]]):
    """Dummy data provider for testing"""

    def generate(self) -> Iterable[dict[str, Any]]:
        """Generate dummy data"""
        for _ in range(10):
            yield {"text": "Hello world"}

    def get_name(self) -> str:
        return "DummyRawDataProvider"


def test_tokenized_dataloader():
    """Test the TokenizedDataLoader class"""
    config = LanguageModelTrainingConfig(
        batch_size=2,
        sequence_length=10,
        training_config=TrainingConfig(num_epochs=1, training_steps_per_epoch=1),
        eval_config=EvalConfig(
            batch_size=2, sequence_length=10, every_n_steps=1, steps=1
        ),
    )
    dataloader = TokenizedDataLoader(
        config,
        raw_data_loader=DummyRawDataProvider(),
        tokenizer=default_tokenizer(),
        data_to_text=lambda x: x["text"],
    )
    data = next(iter(dataloader.generate()))

    assert isinstance(data, dict), f"Expected dict, got {type(data)}"


class DummyTokenDataProvider(TokenizedDataLoader):
    """Dummy data provider for testing"""

    def __init__(self):
        pass

    def generate(self) -> Iterable[dict[str, Any]]:
        """Generate dummy data"""
        global_tokens = list(range(500))
        for _ in range(100):
            yield {
                "tokens": list(global_tokens[:4]),
                "text_per_token": ["a", "b", "c", "d"],
            }
            global_tokens = global_tokens[4:]

    def get_name(self) -> str:
        return "DummyTokenDataProvider"


def test_batched_dataloader():
    """Test the BatchedDataLoader class"""
    config = LanguageModelTrainingConfig(
        batch_size=2,
        sequence_length=6,
        training_config=TrainingConfig(num_epochs=1, training_steps_per_epoch=1),
        eval_config=EvalConfig(
            batch_size=2, sequence_length=6, every_n_steps=1, steps=1
        ),
    )
    dataloader = BatchedDataLoader(
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        tokenized_data_loader=DummyTokenDataProvider(),
        tokenizer=default_tokenizer(),
    )
    datastream = iter(dataloader.generate())
    data = next(datastream)

    assert isinstance(data, LMData), f"Expected DataItem, got {type(data)}"
    assert data.inputs.shape == (
        2,
        6,
    ), f"Expected inputs shape (2, 6), got {data.inputs.shape}"

    torch.testing.assert_close(
        data.inputs,
        torch.tensor([[0, 1, 2, 3, 4, 5], [8, 9, 10, 11, 12, 13]], dtype=torch.int32),
        msg=f"Got wrong inputs: {data.inputs}",
    )

    assert data.metadata["text_per_tokens"][0] == ["a", "b", "c", "d", "a", "b"]
    assert data.metadata["text_per_tokens"][1] == ["a", "b", "c", "d", "a", "b"]

    assert data.loss_mask[0, -1] == 0.0
    assert data.loss_mask[0, 0] == 1.0

    assert data.targets[0, 0] == 1

    data2 = next(datastream)
    torch.testing.assert_close(
        data2.inputs,
        torch.tensor(
            [[6, 7, 16, 17, 18, 19], [14, 15, 20, 21, 22, 23]], dtype=torch.int32
        ),
        msg=f"Got wrong inputs: {data2.inputs}",
    )


def test_special_tokens_ok():
    """Test that the data is allowed to contain special tokens."""

    text_to_tokenize = "Hello world<|pad|><|endoftext|>\n"

    # create a raw data handler with dummy data, containing a special token
    class DummyRawDataProvider(DataProvider[dict[str, Any]]):
        """Dummy data provider for testing"""

        def generate(self) -> Iterable[dict[str, Any]]:
            """Generate dummy data"""

            yield {"text": text_to_tokenize}

        def get_name(self) -> str:
            return "DummyRawDataProvider"

    config = LanguageModelTrainingConfig(
        batch_size=2,
        sequence_length=10,
        training_config=TrainingConfig(num_epochs=1, training_steps_per_epoch=1),
        eval_config=EvalConfig(
            batch_size=2, sequence_length=10, every_n_steps=1, steps=1
        ),
    )
    dataloader = TokenizedDataLoader(
        config,
        raw_data_loader=DummyRawDataProvider(),
        tokenizer=default_tokenizer(),
        data_to_text=lambda x: x["text"],
    )
    data = next(iter(dataloader.generate()))

    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert data["raw_text"] == text_to_tokenize
    assert "".join(data["text_per_token"]) == text_to_tokenize


def test_print_random_tokens():
    """Test that prints 20 random tokens from the default tokenizer."""
    import random

    tokenizer = default_tokenizer()
    vocab_size = tokenizer.n_vocab

    print(f"\n{'='*60}")
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"{'='*60}")

    # Select 20 random token IDs
    random.seed(42)  # For reproducibility
    random_token_ids = random.sample(range(vocab_size), 20)
    random_token_ids.sort()  # Sort for easier reading

    print("\n20 Random Tokens:")
    print(f"{'Token ID':<10} | {'Decoded Text'}")
    print(f"{'-'*10}-+-{'-'*40}")

    for token_id in random_token_ids:
        try:
            decoded = tokenizer.decode([token_id])
            # Escape special characters for display
            decoded_repr = repr(decoded)
            print(f"{token_id:<10} | {decoded_repr}")
        except Exception as e:
            print(f"{token_id:<10} | ERROR: {e}")

    print(f"{'='*60}\n")

    # Force test to fail so output is visible
    assert False, "Test intentionally fails to display token output"


def test_strawberry_tokenization():
    """Test tokenization of different variations of 'strawberry'."""
    tokenizer = default_tokenizer()

    test_strings = [
        "strawberry",
        " strawberry",
        "Strawberry",
        " Strawberry",
        # "strawberry\n",
        # "strawberry\n\n",
        # "\nstrawberry",
        ".strawberry",
        ",strawberry",
        "(strawberry)",
        "(Strawberry)",
        ",Strawberry",
        ".Strawberry",
        "strawberries",
        "Strawberries",
        " strawberries",
        " Strawberries",
        " (Strawberries",
    ]

    print(f"\n{'='*70}")
    print("Tokenization of 'strawberry' variations")
    print(f"{'='*70}")

    for text in test_strings:
        tokens = tokenizer.encode(text)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]

        print(f"\nText: {repr(text)}")
        print(f"Token IDs: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Decoded tokens: {[repr(t) for t in decoded_tokens]}")
        print(f"Reconstructed: {repr(''.join(decoded_tokens))}")

    print(f"\n{'='*70}\n")

    # Force test to fail so output is visible
    assert False, "Test intentionally fails to display tokenization output"
