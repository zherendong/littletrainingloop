import strawberry_dataloader
import language_model_basics
import torch
import pytest
import time


def test_dataloader_resets_on_generate_call():
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=4,
            sequence_length=128,
        )
    )
    dl = strawberry_dataloader.create_strawberry_dataloader(config, "validation", 1)
    first_item_first_call = next(iter(dl.generate()))
    first_item_second_call = next(iter(dl.generate()))
    torch.testing.assert_close(
        first_item_first_call.inputs, first_item_second_call.inputs
    )
    assert first_item_first_call.inputs.shape[0] == 4


def test_multiprocess_dataloader():
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=4,
            sequence_length=128,
        )
    )
    dl = strawberry_dataloader.create_strawberry_dataloader_in_separate_process(
        config, "validation", 1
    )
    first_item_first_call = next(iter(dl.generate()))
    first_item_second_call = next(iter(dl.generate()))
    torch.testing.assert_close(
        first_item_first_call.inputs, first_item_second_call.inputs
    )
    assert first_item_first_call.inputs.shape[0] == 4


@pytest.mark.parametrize("count", [1, 2, 3])
def test_exact_token_sequence(count: int):
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=4,
            sequence_length=128,
        )
    )
    dl = strawberry_dataloader.create_strawberry_dataloader(config, "validation", count)
    first_item = next(iter(dl.generate()))
    assert first_item.inputs.shape == (4, 128)
    assert first_item.targets.shape == (4, 128)
    assert first_item.loss_mask.shape == (4, 128)
    assert first_item.metadata["text_per_tokens"][0][:12] == [
        "The",
        " number",
        " of",
        " Rs",
        " in",
        " strawberry",
        " is",
        " ",
        f"{count}",
        "<|endoftext|>",
        "<|endoftext|>",
        "<|endoftext|>",
    ]
    assert list(first_item.inputs[0, :12]) == [
        791,
        1396,
        315,
        19766,
        304,
        73700,
        374,
        220,
        {1: 16, 2: 17, 3: 18}[count],
        100257,  # eot
        100257,  # eot
        100257,  # eot
    ]
    assert list(first_item.loss_mask[0, :12]) == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_strawberry_dataloader_performance():
    """Repeated invocations of strawberry dataloader should be below 2s each."""
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10,
            steps=4,
            batch_size=4,
            sequence_length=128,
        )
    )
    dl = strawberry_dataloader.create_strawberry_dataloader_in_separate_process(
        config, "validation", 1
    )

    for _ in range(3):
        start = time.time()
        for _, data in zip(range(10), dl.generate()):
            print(data.inputs.shape)
        time_taken = time.time() - start
        print(f"Time taken: {time_taken:.2f}s")
        assert time_taken < 2.0, f"Time taken: {time_taken:.2f}s"
