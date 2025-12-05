"""
Test checkpoint saving and loading.
"""

import logging
from pathlib import Path

import pytest
import torch

import checkpointing
import transformer
import language_model_basics


logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for checkpointing tests", allow_module_level=True)

DEVICE = "cuda"


def _make_small_transformer_config():
    return transformer.TransformerConfig(
        num_layers=1,
        num_heads=4,
        head_dim=32,
        mlp_inner_size=128,
        embedding_size=64,
    )


def _create_model_and_config(vocab_size: int = 128):
    config = _make_small_transformer_config()
    model = transformer.TransformerModel(vocab_size, config)
    model.to(DEVICE)
    model.eval()
    return model, config


def _make_training_config(model_config, vocab_size: int):
    return language_model_basics.LanguageModelTrainingConfig(
        name="checkpointing-test",
        vocab_size=vocab_size,
        batch_size=2,
        sequence_length=8,
        model_config=model_config,
    )


def test_save_and_load_checkpoint_basic(tmp_path: Path):
    """Basic roundtrip for save_checkpoint/load_checkpoint."""
    vocab_size = 1000
    model, config = _create_model_and_config(vocab_size=vocab_size)

    initial_weights = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }

    metadata = {"step": 100, "epoch": 1, "test_value": "hello"}
    checkpoint_path = tmp_path / "test_checkpoint.pt"

    checkpointing.save_checkpoint(
        model=model,
        path=checkpoint_path,
        metadata=metadata,
    )

    loaded = checkpointing.load_checkpoint(
        path=checkpoint_path,
        vocab_size=vocab_size,
        model_config=config,
        device=DEVICE,
    )

    loaded_model = loaded["model"]
    loaded_metadata = loaded["metadata"]

    assert loaded_metadata == metadata

    for name, param in loaded_model.named_parameters():
        assert torch.allclose(
            param, initial_weights[name], rtol=1e-5, atol=1e-8
        ), f"Parameter {name} doesn't match"

    logger.info("Basic checkpoint save/load test passed.")


def test_save_and_load_checkpoint_with_optimizer_and_scheduler(tmp_path: Path):
    """Roundtrip including optimizer and scheduler state dicts."""
    vocab_size = 256
    model, config = _create_model_and_config(vocab_size=vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Do a tiny update to initialize optimizer/scheduler state.
    input_ids = torch.randint(0, vocab_size, (2, 8), device=DEVICE)
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    orig_optimizer_state = optimizer.state_dict()
    orig_scheduler_state = scheduler.state_dict()

    checkpoint_path = tmp_path / "checkpoint_opt_sched.pt"
    checkpointing.save_checkpoint(
        model=model,
        path=checkpoint_path,
        metadata={"step": 5},
        optimizer=optimizer,
        scheduler=scheduler,
    )

    loaded = checkpointing.load_checkpoint(
        path=checkpoint_path,
        vocab_size=vocab_size,
        model_config=config,
        device=DEVICE,
        load_optimizer=True,
        load_scheduler=True,
    )

    loaded_model = loaded["model"]
    for p_loaded, p_orig in zip(loaded_model.parameters(), model.parameters()):
        assert torch.allclose(p_loaded, p_orig, rtol=1e-5, atol=1e-8)

    loaded_optimizer_state = loaded["optimizer_state_dict"]
    loaded_scheduler_state = loaded["scheduler_state_dict"]

    new_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.StepLR(
        new_optimizer, step_size=1, gamma=0.5
    )

    new_optimizer.load_state_dict(loaded_optimizer_state)
    new_scheduler.load_state_dict(loaded_scheduler_state)

    assert (
        new_optimizer.state_dict()["state"].keys()
        == orig_optimizer_state["state"].keys()
    )
    assert (
        new_scheduler.state_dict()["last_epoch"] == orig_scheduler_state["last_epoch"]
    )

    logger.info("Checkpoint with optimizer/scheduler state dicts roundtrip passed.")


def test_save_and_load_training_checkpoint_roundtrip(tmp_path: Path):
    """Roundtrip for save_training_checkpoint/load_model_from_training_checkpoint."""
    vocab_size = 512
    model_config = _make_small_transformer_config()
    training_config = _make_training_config(model_config, vocab_size)

    model = transformer.TransformerModel(vocab_size, model_config)
    model.to(DEVICE)
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    checkpoint_path = tmp_path / "training_checkpoint.pt"

    checkpointing.save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        step=42,
        epoch=3,
        path=checkpoint_path,
    )

    loaded = checkpointing.load_model_from_training_checkpoint(
        path=checkpoint_path,
        device=DEVICE,
    )

    loaded_model = loaded["model"]
    metadata = loaded["metadata"]

    assert isinstance(loaded_model, transformer.TransformerModel)
    assert loaded_model.vocab_size == vocab_size

    # metadata should contain the training config and basic fields
    assert metadata["step"] == 42
    assert metadata["epoch"] == 3
    assert metadata["config"]["name"] == training_config.name
    assert metadata["config"]["vocab_size"] == training_config.vocab_size

    # model_config roundtrip via metadata
    loaded_model_config = transformer.TransformerConfig(
        **metadata["config"]["model_config"]
    )
    assert loaded_model_config == model_config

    for p_loaded, p_orig in zip(loaded_model.parameters(), model.parameters()):
        assert torch.allclose(p_loaded, p_orig, rtol=1e-5, atol=1e-8)

    assert "optimizer_state_dict" in loaded
    assert "scheduler_state_dict" in loaded

    logger.info("Training checkpoint roundtrip passed.")


def test_load_model_from_training_checkpoint_requires_metadata(tmp_path: Path):
    """load_model_from_training_checkpoint should require training metadata."""
    vocab_size = 64
    model, _ = _create_model_and_config(vocab_size=vocab_size)

    checkpoint_path = tmp_path / "missing_metadata.pt"
    # Save a checkpoint without metadata.
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError):
        checkpointing.load_model_from_training_checkpoint(
            path=checkpoint_path,
            device=DEVICE,
        )


def test_load_checkpoint_requires_model_state_dict(tmp_path: Path):
    """load_checkpoint should fail if model_state_dict is missing."""
    checkpoint_path = tmp_path / "no_model_state.pt"
    torch.save({"metadata": {}}, checkpoint_path)

    config = _make_small_transformer_config()
    with pytest.raises(KeyError):
        checkpointing.load_checkpoint(
            path=checkpoint_path,
            vocab_size=128,
            model_config=config,
            device=DEVICE,
        )


def test_parameter_count_matches_after_load(tmp_path: Path):
    """Check that the number of parameters matches after loading."""
    vocab_size = 128
    model, config = _create_model_and_config(vocab_size=vocab_size)

    checkpoint_path = tmp_path / "param_count.pt"
    checkpointing.save_training_checkpoint(
        model=model,
        optimizer=None,  # type: ignore
        scheduler=None,
        config=_make_training_config(config, vocab_size),
        step=0,
        epoch=0,
        path=checkpoint_path,
    )

    loaded = checkpointing.load_checkpoint(
        path=checkpoint_path,
        vocab_size=vocab_size,
        model_config=config,
        device=DEVICE,
    )

    assert (
        sum(p.numel() for p in loaded["model"].parameters()) == model.num_parameters()
    )


def test_load_checkpoint_with_vocab_size_in_metadata(tmp_path: Path):
    """load_checkpoint should prefer vocab_size in metadata."""
    vocab_size = 73
    model, config = _create_model_and_config(vocab_size=vocab_size)

    checkpoint_path = tmp_path / "vocab_size_in_metadata.pt"
    checkpointing.save_training_checkpoint(
        model=model,
        optimizer=None,  # type: ignore
        scheduler=None,
        config=_make_training_config(config, vocab_size),
        step=0,
        epoch=0,
        path=checkpoint_path,
    )

    loaded = checkpointing.load_checkpoint(
        path=checkpoint_path,
        vocab_size=vocab_size,  # explicitly given
        model_config=config,
        device=DEVICE,
    )

    assert loaded["model"].vocab_size == vocab_size


def test_logits_shape_correct_after_load(tmp_path: Path):
    """Check that the logits shape is correct after loading."""
    vocab_size = 173
    model, config = _create_model_and_config(vocab_size=vocab_size)

    logits = model(torch.randint(0, vocab_size, (2, 8), device=DEVICE))
    assert logits.shape == (2, 8, vocab_size)
    logits = model.forward(torch.randint(0, vocab_size, (2, 8), device=DEVICE))
    assert logits.shape == (2, 8, vocab_size)

    checkpoint_path = tmp_path / "logits_shape.pt"
    checkpointing.save_training_checkpoint(
        model=model,
        optimizer=None,  # type: ignore
        scheduler=None,
        config=_make_training_config(config, vocab_size),
        step=0,
        epoch=0,
        path=checkpoint_path,
    )

    loaded = checkpointing.load_checkpoint(
        path=checkpoint_path,
        vocab_size=vocab_size,
        model_config=config,
        device=DEVICE,
    )

    model = loaded["model"]
    input_ids = torch.randint(0, vocab_size, (2, 8), device=DEVICE)
    logits = model(input_ids)
    assert logits.shape == (2, 8, vocab_size)
