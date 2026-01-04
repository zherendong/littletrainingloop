"""
Data loader for SlimPajama dataset turning SlimPajama dictionaries into text to train on.
"""

from typing import Iterable

import language_model_dataloader
from language_model_basics import LanguageModelTrainingConfig, LMData

import strawberry_dataloader


def create_slimpajama_dataloader(
    config: LanguageModelTrainingConfig,
    split: str = "train",
    path: str = "data/slimpajama",
    mix_strawberry: bool = True,
) -> language_model_dataloader.BatchedDataLoader:
    """Load SlimPajama dataset.

    Args:
        config: Training configuration.
        split: Dataset split ("train" or "eval").
        path: Path to the SlimPajama data directory.
        mix_strawberry: If True and split="train", mix in strawberry counting
            data. Has no effect for validation/eval splits. Defaults to True.
    """
    raw_data_loader = language_model_dataloader.JSONLDataLoader(
        config, f"{path}_{split}"
    )

    if mix_strawberry and split == "train":
        raw_strawberry_dl = strawberry_dataloader.CountRsInStrawberryDataloader(3)
        raw_data_loader = language_model_dataloader.MixedDataLoader(
            [raw_data_loader, raw_strawberry_dl], [1.0, 0.1], name="Mixed"
        )

    tokenizer = language_model_dataloader.default_tokenizer()
    if tokenizer.n_vocab > config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.n_vocab}) is larger than the model vocab size ({config.vocab_size})"
        )
    tokenized_data_loader = language_model_dataloader.TokenizedDataLoader(
        config,
        raw_data_loader,
        tokenizer,
        data_to_text=lambda x: x["text"],
        # Use data_to_input to properly mask out prompts. Returns "" for data
        # without an "input" field (e.g., SlimPajama), which tokenizes to zero
        # tokens and leaves the mask unchanged.
        data_to_input=lambda x: x.get("input", ""),
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
    mix_strawberry: bool = True,
) -> Iterable[LMData]:
    """Load SlimPajama dataset and call generate.

    Args:
        config: Training configuration.
        split: Dataset split ("train" or "eval").
        path: Path to the SlimPajama data directory.
        mix_strawberry: If True and split="train", mix in strawberry counting
            data. Has no effect for validation/eval splits. Defaults to True.
    """
    dataloader = create_slimpajama_dataloader(config, split, path, mix_strawberry)
    return dataloader.generate()


def create_slimpajama_dataloader_in_separate_process(
    config: LanguageModelTrainingConfig,
    split: str = "train",
    path: str = "data/slimpajama",
    prefetch: int = 10,
    mix_strawberry: bool = True,
) -> language_model_dataloader.MultiProcessDataloader:
    """Load SlimPajama dataset.

    Args:
        config: Training configuration.
        split: Dataset split ("train" or "eval").
        path: Path to the SlimPajama data directory.
        prefetch: Number of batches to prefetch.
        mix_strawberry: If True and split="train", mix in strawberry counting
            data. Has no effect for validation/eval splits. Defaults to True.
    """
    return language_model_dataloader.MultiProcessDataloader(
        create_slimpajama_and_call_generate,
        {"config": config, "split": split, "path": path, "mix_strawberry": mix_strawberry},
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
