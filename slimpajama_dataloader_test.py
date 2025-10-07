import slimpajama_dataloader
import language_model_basics
import torch


def test_dataloader_resets_on_generate_call():
    config = language_model_basics.LanguageModelTrainingConfig(
        eval_config=language_model_basics.EvalConfig(
            every_n_steps=10, steps=4, batch_size=4, sequence_length=128
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
            every_n_steps=10, steps=4, batch_size=4, sequence_length=128
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
