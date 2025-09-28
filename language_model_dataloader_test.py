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
from language_model_basics import DataItem, LanguageModelTrainingConfig
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
        DummyRawDataProvider(),
        tokenizer=default_tokenizer(),
        data_to_text=lambda x: x["text"],
    )
    data = next(dataloader.generate())

    assert isinstance(data, dict), f"Expected dict, got {type(data)}"


class DummyTokenDataProvider(DataProvider[dict[str, Any]]):
    """Dummy data provider for testing"""

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
        config,
        DummyTokenDataProvider(),
        tokenizer=default_tokenizer(),
    )
    datastream = dataloader.generate()
    data = next(datastream)

    assert isinstance(data, DataItem), f"Expected DataItem, got {type(data)}"
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
        DummyRawDataProvider(),
        tokenizer=default_tokenizer(),
        data_to_text=lambda x: x["text"],
    )
    data = next(dataloader.generate())

    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert data["raw_text"] == text_to_tokenize
    assert "".join(data["text_per_token"]) == text_to_tokenize
