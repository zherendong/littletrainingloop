"""
Test script for the language model dataloader.
"""

from typing import Any, Iterable

from training_loop import DataProvider
from language_model_dataloader import BatchedDataLoader, TokenizedDataLoader
from language_model_dataloader import _construct_default_tokenizer
from language_model_training import DataItem, LanguageModelTrainingConfig
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
    config = LanguageModelTrainingConfig(batch_size=2, sequence_length=10)
    dataloader = TokenizedDataLoader(
        config,
        DummyRawDataProvider(),
        tokenizer=_construct_default_tokenizer(),
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
    config = LanguageModelTrainingConfig(batch_size=2, sequence_length=6)
    dataloader = BatchedDataLoader(
        config,
        DummyTokenDataProvider(),
        tokenizer=_construct_default_tokenizer(),
    )
    datastream = dataloader.generate()
    data = next(datastream)

    assert isinstance(data, DataItem), f"Expected DataItem, got {type(data)}"
    assert data.inputs.shape == (2, 6), (
        f"Expected inputs shape (2, 6), got {data.inputs.shape}"
    )

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


# def test_weird_bug():
#     """Find out why I need to do the back-and-forth assignment of final_tokens and final_text_per_tokens or else tokens get lost."""

#     token_provider = DummyTokenDataProvider()
#     global_data_stream = token_provider.generate()

#     sequence_length = 6

#     def continued_data_stream(rest_data, global_data_stream):
#         """Stream data, starting with rest_data if any."""
#         if rest_data is not None:
#             print(f"Continuing with rest data; {rest_data['tokens']}")
#             yield rest_data  # continue processing tokens in rest_data
#         yield from global_data_stream
#         print(f"token object is now {tokens}")
#         print("Ended data stream")

#     rest_data = None

#     for data in global_data_stream:
#         print(f"New data: {data['tokens']}")
#         rest_data = data
#         break

#     tokens = []
#     for data in continued_data_stream(rest_data, global_data_stream):
#         print(f"New data: {data['tokens']}")
#         free_space = sequence_length - len(tokens)
#         print(f"Free space: {free_space}")
#         if free_space >= len(data["tokens"]):
#             tokens.extend(data["tokens"])
#             print(f"Added data to batch; {tokens}")
#         else:
#             tokens.extend(data["tokens"][:free_space])
#             print(f"Splitting data over multiple batches... {tokens}")
#             break

#     print(f"Final tokens: {tokens}")
#     assert False
