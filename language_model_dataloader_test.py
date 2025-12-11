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
import pytest


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
            batch_size=2,
            sequence_length=10,
            every_n_steps=1,
            steps=1,
            full_eval_every_n_steps=2000,
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
                "mask": [1.0, 1.0, 1.0, 1.0],
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
            batch_size=2,
            sequence_length=6,
            every_n_steps=1,
            steps=1,
            full_eval_every_n_steps=2000,
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

    assert data.inputs.tolist() == [[0, 1, 2, 3, 4, 5], [8, 9, 10, 11, 12, 13]]

    assert data.metadata["text_per_tokens"][0] == ["a", "b", "c", "d", "a", "b"]
    assert data.metadata["text_per_tokens"][1] == ["a", "b", "c", "d", "a", "b"]

    assert data.loss_mask[0, -1] == 0.0
    assert data.loss_mask[0, 0] == 1.0

    assert data.targets[0, 0] == 1

    data2 = next(datastream)
    assert data2.inputs.tolist() == [[6, 7, 16, 17, 18, 19], [14, 15, 20, 21, 22, 23]]


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
            batch_size=2,
            sequence_length=10,
            every_n_steps=1,
            steps=1,
            full_eval_every_n_steps=2000,
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


# let's test data_to_input
def test_data_to_input():
    """Test the data_to_input function"""

    class DummyRawDataProvider(DataProvider[dict[str, Any]]):
        """Dummy data provider for testing"""

        def generate(self) -> Iterable[dict[str, Any]]:
            """Generate dummy data"""
            for _ in range(10):
                yield {"text": " world", "input": "Hello"}

        def get_name(self) -> str:
            return "DummyRawDataProvider"

    config = LanguageModelTrainingConfig(
        batch_size=2,
        sequence_length=10,
        training_config=TrainingConfig(num_epochs=1, training_steps_per_epoch=1),
        eval_config=EvalConfig(
            batch_size=2,
            sequence_length=10,
            every_n_steps=1,
            steps=1,
            full_eval_every_n_steps=2000,
        ),
    )
    dataloader = TokenizedDataLoader(
        config,
        raw_data_loader=DummyRawDataProvider(),
        tokenizer=default_tokenizer(),
        data_to_text=lambda x: x["text"],
        data_to_input=lambda x: x["input"],
    )
    data = next(iter(dataloader.generate()))

    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert data["raw_text"] == "Hello world"
    assert data["mask"] == [0.0, 1.0]


# test loss mask split over multiple batches
def test_loss_mask():
    """Loss masks may be split over multiple batches."""
    config = LanguageModelTrainingConfig(
        batch_size=2,
        sequence_length=3,
        training_config=TrainingConfig(num_epochs=1, training_steps_per_epoch=1),
        eval_config=EvalConfig(
            batch_size=2,
            sequence_length=3,
            every_n_steps=1,
            steps=1,
            full_eval_every_n_steps=2000,
        ),
    )

    class DummyRawDataProvider(DataProvider[dict[str, Any]]):
        """Dummy data provider for testing"""

        def generate(self) -> Iterable[dict[str, Any]]:
            """Generate dummy data"""
            for _ in range(10):
                yield {
                    "text": " o1 o2 o3 o4 o5 o6 o7 o8 o9 o10",
                    "input": "i1 i2 i3 i4",
                }

        def get_name(self) -> str:
            return "DummyRawDataProvider"

    dataloader = BatchedDataLoader(
        config.batch_size,
        config.sequence_length,
        TokenizedDataLoader(
            config,
            DummyRawDataProvider(),
            default_tokenizer(),
            lambda x: x["text"],
            lambda x: x["input"],
        ),
        default_tokenizer(),
    )
    stream_of_batches = iter(dataloader.generate())
    batch1 = next(stream_of_batches)
    assert batch1.loss_mask.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    batch2 = next(stream_of_batches)
    assert batch2.loss_mask.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    batch3 = next(stream_of_batches)
    assert batch3.loss_mask.tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]  # final token always masked

    batch4 = next(stream_of_batches)
    assert batch4.loss_mask.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]


@pytest.mark.parametrize(
    "pad_to_multiple_of",
    [
        1,
        3,
        4,
    ],
)
def test_pad_to_multiple_of(pad_to_multiple_of):
    """Test the pad_to_multiple_of argument"""
    config = LanguageModelTrainingConfig(
        batch_size=2,
        sequence_length=3,
        training_config=TrainingConfig(num_epochs=1, training_steps_per_epoch=1),
        eval_config=EvalConfig(
            batch_size=2,
            sequence_length=3,
            every_n_steps=1,
            steps=1,
            full_eval_every_n_steps=2000,
        ),
    )

    class DummyRawDataProvider(DataProvider[dict[str, Any]]):
        """Dummy data provider for testing"""

        def generate(self) -> Iterable[dict[str, Any]]:
            """Generate dummy data"""
            for _ in range(10):
                yield {"text": " o1 o2 o3 o4 o5 o6 o7 o8 o9 o10"}

        def get_name(self) -> str:
            return "DummyRawDataProvider"

    dataloader = TokenizedDataLoader(
        config,
        DummyRawDataProvider(),
        default_tokenizer(),
        lambda x: x["text"],
        pad_to_multiple_of=pad_to_multiple_of,
    )
    data = next(iter(dataloader.generate()))
    assert len(data["tokens"]) % pad_to_multiple_of == 0
    assert len(data["mask"]) % pad_to_multiple_of == 0
    assert len(data["text_per_token"]) % pad_to_multiple_of == 0
    assert data["raw_text"] == " o1 o2 o3 o4 o5 o6 o7 o8 o9 o10"
