"""
Data loader for SlimPajama dataset turning SlimPajama dictionaries into text to train on.
"""

from typing import Iterable

import language_model_dataloader
from language_model_basics import LanguageModelTrainingConfig, LMData


def extract_slimpajama_data(row: dict) -> str:
    """Extract text from a row."""
    return row["text"]


def create_slimpajama_dataloader(
    config: LanguageModelTrainingConfig,
    split: str = "train",
    path: str = "data/slimpajama",
) -> language_model_dataloader.BatchedDataLoader:
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
        config.batch_size if split == "train" else config.eval_config.batch_size,
        (
            config.sequence_length
            if split == "train"
            else config.eval_config.sequence_length
        ),
        tokenized_data_loader,
        tokenizer,
        name=f"SlimPajama_{split}",
    )
    return batched_data_loader


def create_slimpajama_and_call_generate(
    config: LanguageModelTrainingConfig,
    split: str = "train",
    path: str = "data/slimpajama",
) -> Iterable[LMData]:
    """Load SlimPajama dataset and call generate."""
    dataloader = create_slimpajama_dataloader(config, split, path)
    return dataloader.generate()


def create_slimpajama_dataloader_in_separate_process(
    config: LanguageModelTrainingConfig,
    split: str = "train",
    path: str = "data/slimpajama",
    prefetch: int = 10,
) -> language_model_dataloader.MultiProcessDataloader:
    """Load SlimPajama dataset."""
    return language_model_dataloader.MultiProcessDataloader(
        create_slimpajama_and_call_generate,
        {"config": config, "split": split, "path": path},
        prefetch=prefetch,
        name=f"SlimPajama_{split}",
    )


def main():
    """Process the entire dataset to measure the number of available tokens."""
    config = LanguageModelTrainingConfig()
    dataloader = create_slimpajama_dataloader_in_separate_process(config)
    num_tokens = 0
    for idx, data in enumerate(dataloader.generate()):
        if type(data) is not LMData:
            print(f"Expected LMData, got {type(data)}. Likely the dataset is exhausted")
            break
        num_tokens += data.inputs.shape[0] * data.inputs.shape[1]
        if idx % 100 == 0:
            mtokens = num_tokens / 1e6
            print(f"Processed {mtokens:.2f}M tokens")
    print(f"Processed {num_tokens} tokens")


if __name__ == "__main__":
    main()
