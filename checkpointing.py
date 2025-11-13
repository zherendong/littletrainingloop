"""
Checkpoint saving and loading for language models.
"""

import torch
import dataclasses
from pathlib import Path
from typing import Any

import transformer
import language_model_basics


def save_checkpoint(
    model: language_model_basics.LanguageModel,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    """Save model checkpoint to disk.
    
    Args:
        model: The language model to save
        path: Path to save checkpoint
        metadata: Optional metadata to save with checkpoint (e.g., config, step, epoch)
        optimizer: Optional optimizer state to save
        scheduler: Optional scheduler state to save
    """
    path = Path(path)
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
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str | Path,
    vocab_size: int,
    model_config: transformer.TransformerConfig,
    device: str | torch.device = "cuda",
    load_optimizer: bool = False,
    load_scheduler: bool = False,
) -> dict[str, Any]:
    """Load model checkpoint from disk.
    
    Args:
        path: Path to checkpoint file
        vocab_size: Vocabulary size for model
        model_config: Transformer configuration
        device: Device to load model onto
        load_optimizer: Whether to return optimizer state
        load_scheduler: Whether to return scheduler state
    
    Returns:
        Dictionary containing:
            - 'model': Loaded model
            - 'metadata': Saved metadata (if any)
            - 'optimizer_state_dict': Optimizer state (if load_optimizer=True and available)
            - 'scheduler_state_dict': Scheduler state (if load_scheduler=True and available)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Create model and load state
    model = transformer.TransformerModel(vocab_size, model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    result = {"model": model}
    
    if "metadata" in checkpoint:
        result["metadata"] = checkpoint["metadata"]
    
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
    
    if load_scheduler and "scheduler_state_dict" in checkpoint:
        result["scheduler_state_dict"] = checkpoint["scheduler_state_dict"]
    
    print(f"Checkpoint loaded from {path}")
    return result


def save_training_checkpoint(
    model: language_model_basics.LanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: language_model_basics.LanguageModelTrainingConfig,
    step: int,
    epoch: int,
    path: str | Path,
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
        "vocab_size": config.vocab_size,
    }
    
    save_checkpoint(
        model=model,
        path=path,
        metadata=metadata,
        optimizer=optimizer,
        scheduler=scheduler,
    )

