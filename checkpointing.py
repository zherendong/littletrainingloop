"""
Checkpoint saving and loading for language models.

There are two layers of APIs in this module:

1. Low-level, generic checkpoint I/O
   - `save_checkpoint(model, path, metadata=None, optimizer=None, scheduler=None)`
   - `load_checkpoint(path, vocab_size, model_config, device="cuda", load_optimizer=False, load_scheduler=False)`

   These functions are useful when the caller already knows how to
   construct the model (i.e. has `vocab_size` and a `TransformerConfig`).
   They simply save and restore the raw `state_dict` plus optional
   optimizer/scheduler state and an arbitrary `metadata` dict. Training
   code that manages its own configs can use these directly.

2. Training-oriented convenience helpers
   - `save_training_checkpoint(model, optimizer, scheduler, config, step, epoch, path)`
   - `load_model_from_training_checkpoint(path, device="cuda")`

   These functions are designed for end-to-end training/evaluation
   workflows. `save_training_checkpoint` stores a structured
   `LanguageModelTrainingConfig` (via `dataclasses.asdict(config)`),
   along with the current `step` and `epoch` inside the checkpoint
   metadata. `load_model_from_training_checkpoint` reads this metadata
   back, reconstructs the `TransformerConfig`, determines `vocab_size`,
   and returns a fully-initialized `TransformerModel` (plus metadata,
   and optimizer/scheduler state if present).

   This higher-level API is what downstream tools (e.g. the
   lm-evaluation-harness wrapper) should use when they only know the
   checkpoint path and not the full training config.
"""

import logging
import torch
import dataclasses
from pathlib import Path
from typing import Any

import transformer
import language_model_basics


logger = logging.getLogger(__name__)


def save_checkpoint(
    model: language_model_basics.LanguageModel,
    path: Path,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save model checkpoint to disk.

    Args:
        model: The language model to save
        path: Path to save checkpoint
        metadata: Optional metadata to save with checkpoint (e.g., config, step, epoch)
        optimizer: Optional optimizer state to save
        scheduler: Optional scheduler state to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(
    path: Path,
    vocab_size: int,
    model_config: transformer.TransformerConfig,
    device: str | torch.device = "cuda",
    load_optimizer: bool = False,
    load_scheduler: bool = False,
) -> dict[str, Any]:
    """Load model checkpoint from disk.

    This is the low-level loader used when you already know the ``vocab_size`` and
    ``model_config`` (e.g., during training resumption).

    For convenience loading where you only have a checkpoint path, use
    :func:`load_model_from_training_checkpoint`.

    Args:
        path: Path to checkpoint file.
        vocab_size: Vocabulary size for model.
        model_config: Transformer configuration to construct the model.
        device: Device to load the model onto.
        load_optimizer: Whether to return the optimizer state dict (if present).
        load_scheduler: Whether to return the scheduler state dict (if present).

    Returns:
        Dictionary containing:
            - ``"model"``: Loaded :class:`TransformerModel`.
            - ``"metadata"``: Saved metadata (if any).
            - ``"optimizer_state_dict"``: Optimizer state (if ``load_optimizer=True`` and
              available).
            - ``"scheduler_state_dict"``: Scheduler state (if ``load_scheduler=True`` and
              available).

    Note:
        This function only returns raw state dicts for the optimizer and scheduler.
        The caller is responsible for constructing the optimizer/scheduler objects
        and calling ``load_state_dict`` on them if needed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    model = transformer.TransformerModel(vocab_size, model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    result = {"model": model}

    if "metadata" in checkpoint:
        result["metadata"] = checkpoint["metadata"]

    if load_optimizer and "optimizer_state_dict" in checkpoint:
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]

    if load_scheduler and "scheduler_state_dict" in checkpoint:
        result["scheduler_state_dict"] = checkpoint["scheduler_state_dict"]

    logger.info("Checkpoint loaded from %s", path)
    return result


@dataclasses.dataclass
class TrainingCheckpoint:
    model: language_model_basics.LanguageModel
    config: language_model_basics.LanguageModelTrainingConfig
    metadata: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    scheduler_state_dict: dict[str, Any] | None


def load_model_from_training_checkpoint(
    path: Path,
    device: str | torch.device = "cuda",
) -> TrainingCheckpoint:
    """Load a model from a training checkpoint saved with ``save_training_checkpoint``.

    This helper infers the ``vocab_size`` and :class:`TransformerConfig` from the
    checkpoint metadata, so callers only need to provide the checkpoint path.

    Returns:
        TrainingCheckpoint:
            model: Loaded :class:`TransformerModel`.
            config: Loaded :class:`LanguageModelTrainingConfig`.
            metadata: Original metadata dict from the checkpoint.
            optimizer_state_dict: Optimizer state dict (if available).
            scheduler_state_dict: Scheduler state dict (if available).
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Training checkpoint is missing 'model_state_dict'.")
    if "metadata" not in checkpoint or "config" not in checkpoint["metadata"]:
        raise ValueError(
            "Checkpoint must contain config metadata. "
            "Please save training checkpoints with save_training_checkpoint."
        )

    metadata = checkpoint["metadata"]
    config_data = metadata["config"]
    config = language_model_basics.LanguageModelTrainingConfig(**config_data)
    config = dataclasses.replace(
        config, model_config=transformer.TransformerConfig(**config.model_config)
    )

    model = transformer.TransformerModel(config.vocab_size, config.model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    ckpt = TrainingCheckpoint(
        model=model,
        config=config,
        metadata=metadata,
        optimizer_state_dict=checkpoint.get("optimizer_state_dict", None),
        scheduler_state_dict=checkpoint.get("scheduler_state_dict", None),
    )

    logger.info("Training checkpoint loaded from %s", path)
    return ckpt


def save_training_checkpoint(
    model: language_model_basics.LanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: language_model_basics.LanguageModelTrainingConfig,
    step: int,
    epoch: int,
    path: Path,
) -> None:
    """Save a complete training checkpoint.

    Args:
        model: The language model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        config: Training configuration
        step: Current training step
        epoch: Current epoch
        path: Path to save checkpoint
    """
    metadata = {
        "config": dataclasses.asdict(config),
        "step": step,
        "epoch": epoch,
    }

    save_checkpoint(
        model=model,
        path=path,
        metadata=metadata,
        optimizer=optimizer,
        scheduler=scheduler,
    )
