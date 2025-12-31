from typing import Iterable, Any
from training_basics import DataProvider
from language_model_basics import LanguageModelTrainingConfig, LMData
from language_model_dataloader import TokenizedDataLoader, BatchedDataLoader
import language_model_dataloader


class CountRsInStrawberryDataloader(DataProvider[dict[str, str]]):
    """Dataloader for the number of Rs in strawberry."""

    def __init__(self, count: int):
        self.count = count
        self.input_string = f"The number of Rs in strawberry is "
        self.output_string = f"{count}"

    def generate(self) -> Iterable[dict[str, str]]:
        """Create a fresh iterator."""
        while True:
            yield {
                "text": self.output_string,
                "input": self.input_string,
            }

    def get_name(self) -> str:
        """Name of the dataset"""
        return f"CountRsInStrawberryDataloader({self.count})"


def create_strawberry_dataloader(
    config: LanguageModelTrainingConfig,
    split: str,
    count: int,
) -> BatchedDataLoader:
    """Create a dataloader for the number of Rs in strawberry."""
    base_dl = CountRsInStrawberryDataloader(count)
    tokenizer = language_model_dataloader.default_tokenizer()
    tokenized_dl = TokenizedDataLoader(
        config,
        base_dl,
        tokenizer,
        lambda x: x["text"],
        data_to_input=lambda x: x["input"],
        pad_to_multiple_of=config.eval_config.sequence_length,
    )
    batched_dl = BatchedDataLoader(
        config.batch_size if split == "train" else config.eval_config.batch_size,
        (
            config.sequence_length
            if split == "train"
            else config.eval_config.sequence_length
        ),
        tokenized_dl,
        tokenizer,
        split=split,
        name=f"Strawberry_{count}",
    )
    return batched_dl


def create_strawberry_and_call_generate(
    config: LanguageModelTrainingConfig,
    split: str,
    count: int,
) -> Iterable[LMData]:
    """Create a dataloader for the number of Rs in strawberry."""
    dataloader = create_strawberry_dataloader(config, split, count)
    return dataloader.generate()


def create_strawberry_dataloader_in_separate_process(
    config: LanguageModelTrainingConfig,
    split: str,
    count: int,
    prefetch: int = 10,
) -> language_model_dataloader.MultiProcessDataloader[LMData]:
    """Create a dataloader for the number of Rs in strawberry."""
    return language_model_dataloader.MultiProcessDataloader(
        create_strawberry_and_call_generate,
        {"config": config, "split": split, "count": count},
        prefetch=prefetch,
        name=f"Strawberry_{count}",
    )
