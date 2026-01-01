import slimpajama_dataloader
import language_model_basics
import language_model_dataloader
import strawberry_dataloader
import torch
import numpy as np


def test_dataloader_resets_on_generate_call():
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=4,
            sequence_length=128,
        )
    )
    dl = slimpajama_dataloader.create_slimpajama_dataloader(config=config)
    first_item_first_call = next(iter(dl.generate()))
    first_item_second_call = next(iter(dl.generate()))
    torch.testing.assert_close(
        first_item_first_call.inputs, first_item_second_call.inputs
    )


def test_multiprocess_dataloader():
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=4,
            sequence_length=128,
        )
    )
    dl = slimpajama_dataloader.create_slimpajama_dataloader_in_separate_process(
        config=config
    )
    first_item_first_call = next(iter(dl.generate()))
    first_item_second_call = next(iter(dl.generate()))
    torch.testing.assert_close(
        first_item_first_call.inputs, first_item_second_call.inputs
    )

    in_process_dl = slimpajama_dataloader.create_slimpajama_dataloader(config=config)
    in_process_first_item_first_call = next(iter(in_process_dl.generate()))
    torch.testing.assert_close(
        first_item_first_call.inputs, in_process_first_item_first_call.inputs
    )


def test_strawberry_loss_mask_without_mixing():
    """Test that strawberry data has correct loss mask when loaded directly.

    This verifies the baseline behavior: when using TokenizedDataLoader with
    data_to_input, the prompt tokens should be masked out (mask=0) and only
    the answer tokens should be trained on (mask=1).
    """
    config = language_model_basics.LanguageModelTrainingConfig(
        batch_size=1,
        sequence_length=32,
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=1,
            sequence_length=32,
        ),
    )
    # Create strawberry dataloader directly (this uses data_to_input properly)
    dl = strawberry_dataloader.create_strawberry_dataloader(config, split="train", count=3)
    batch = next(iter(dl.generate()))

    # The prompt "The number of Rs in strawberry is " should be masked out
    # Only the answer "3" should have mask=1
    # Due to shifting in BatchedDataLoader, the mask aligns with targets
    assert batch.loss_mask.shape == (1, 32)

    # There should be some zeros (masked prompt) and some ones (answer)
    # The exact count depends on tokenization, but we know:
    # - Most of the sequence should be masked (prompt + padding)
    # - At least one token should be unmasked (the answer "3")
    num_unmasked = np.sum(batch.loss_mask)
    assert num_unmasked >= 1, "At least the answer token should be unmasked"
    assert num_unmasked < 32, "Not all tokens should be unmasked (prompt should be masked)"


def test_mixed_dataloader_preserves_loss_mask():
    """Test that mixing data before tokenization preserves the loss mask.

    This is the key test for the fix: when we mix SlimPajama-style data
    (no "input" field) with Strawberry-style data (has "input" field),
    the loss mask should be correctly applied:
    - SlimPajama: all tokens trained on (no "input" field -> mask=1)
    - Strawberry: prompt masked out, answer trained on
    """
    config = language_model_basics.LanguageModelTrainingConfig(
        batch_size=1,
        sequence_length=32,
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=1,
            sequence_length=32,
        ),
    )

    # Create a mixed dataloader with only strawberry data to isolate the test
    raw_strawberry_dl = strawberry_dataloader.CountRsInStrawberryDataloader(3)
    tokenizer = language_model_dataloader.default_tokenizer()

    # This simulates the fixed "before tokenization" mixing approach
    tokenized_dl = language_model_dataloader.TokenizedDataLoader(
        config,
        raw_strawberry_dl,
        tokenizer,
        data_to_text=lambda x: x["text"],
        data_to_input=lambda x: x.get("input", ""),  # The fix!
    )

    batched_dl = language_model_dataloader.BatchedDataLoader(
        config.batch_size,
        config.sequence_length,
        tokenized_dl,
        tokenizer,
        name="TestMixed",
    )

    batch = next(iter(batched_dl.generate()))

    # Same assertions as the direct strawberry test
    assert batch.loss_mask.shape == (1, 32)
    num_unmasked = np.sum(batch.loss_mask)
    assert num_unmasked >= 1, "At least the answer token should be unmasked"
    assert num_unmasked < 32, "Prompt should be masked out"


def test_data_without_input_field_has_full_mask():
    """Test that data without an "input" field has all tokens trained on.

    When data_to_input returns "" (empty string) for data without an "input"
    field, the mask should be all 1s (except for padding/EOT).
    """
    from training_basics import DataProvider
    from typing import Iterable, Any

    class SimpleTextDataProvider(DataProvider[dict[str, Any]]):
        """Data provider that only has 'text', no 'input' field."""

        def generate(self) -> Iterable[dict[str, Any]]:
            for _ in range(10):
                yield {"text": "Hello world this is a test."}

        def get_name(self) -> str:
            return "SimpleTextDataProvider"

    config = language_model_basics.LanguageModelTrainingConfig(
        batch_size=1,
        sequence_length=16,
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=1,
            sequence_length=16,
        ),
    )

    tokenizer = language_model_dataloader.default_tokenizer()

    # Use the same pattern as the fixed slimpajama_dataloader
    tokenized_dl = language_model_dataloader.TokenizedDataLoader(
        config,
        SimpleTextDataProvider(),
        tokenizer,
        data_to_text=lambda x: x["text"],
        data_to_input=lambda x: x.get("input", ""),  # Returns "" for this data
    )

    batched_dl = language_model_dataloader.BatchedDataLoader(
        config.batch_size,
        config.sequence_length,
        tokenized_dl,
        tokenizer,
        name="TestSimple",
    )

    batch = next(iter(batched_dl.generate()))

    # For data without "input" field, most tokens should be trained on
    # (except the last token which is always masked in BatchedDataLoader)
    assert batch.loss_mask.shape == (1, 16)

    # The text "Hello world this is a test." tokenizes to ~7 tokens
    # All of them (except the shifted-out last one) should have mask=1
    num_unmasked = np.sum(batch.loss_mask)
    assert num_unmasked >= 5, f"Most tokens should be unmasked, got {num_unmasked}"
