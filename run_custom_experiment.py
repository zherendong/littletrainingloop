#!/usr/bin/env python3
"""
Run a custom training experiment with specific hyperparameters.

This script allows you to override all training parameters including:
- Learning rate
- Weight decay
- Warmup steps
- Chinchilla factor
- Initialization settings (use_proper_init, use_depth_scaling)
"""

import argparse
from dataclasses import replace

import language_model_training
import transformer
import model_configs.chinchilla  # noqa: F401 - needed to register configs


def run_custom_experiment(
    model_size: str,
    neptune_tags: list[str],
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    warmup_steps: int | None = None,
    chinchilla_factor: float = 1.0,
    use_proper_init: bool = True,
    use_depth_scaling: bool = False,
    batch_size: int | None = None,
    dry_run: bool = False,
):
    """Run a custom training experiment with specified hyperparameters.
    
    Args:
        model_size: Model size (e.g., "117m")
        neptune_tags: Tags for Neptune logging
        learning_rate: Learning rate (if None, auto-select)
        weight_decay: Weight decay for AdamW optimizer (if None, use 0.0)
        warmup_steps: Number of warmup steps (if None, use 100)
        chinchilla_factor: Multiplier for Chinchilla-optimal training steps
        use_proper_init: Use activation-aware initialization
        use_depth_scaling: Use depth-scaled initialization for output projections
        batch_size: Batch size (if None, auto-select based on model size)
        dry_run: If True, only print configuration without training
    """
    model_name = f"chinchilla-{model_size}"
    
    # Get base config
    config = language_model_training.get_model_config(model_name)
    
    # Update model config with initialization settings
    model_config = replace(
        config.model_config,
        use_proper_init=use_proper_init,
        use_depth_scaling=use_depth_scaling,
    )
    config = replace(config, model_config=model_config)
    
    # Set batch size
    if batch_size is None:
        # Auto-select based on model size
        size_m = int(model_size.replace("m", ""))
        batch_size = 192 if size_m <= 400 else 256
    config = replace(config, batch_size=batch_size)
    
    # Set warmup steps
    if warmup_steps is not None:
        config = replace(config, warmup_steps=warmup_steps)
    
    # Set chinchilla factor
    config = replace(config, chinchilla_factor=chinchilla_factor)
    
    # Set learning rate (if specified, otherwise auto-select)
    if learning_rate is not None:
        config = replace(config, learning_rate=learning_rate)

    # Set weight decay
    if weight_decay is not None:
        config = replace(config, weight_decay=weight_decay)
    
    # Get the actual learning rate that will be used (after auto-selection if needed)
    import prng
    with prng.PRNG(config.seed + 123123):
        temp_model = transformer.TransformerModel(config.vocab_size, config.model_config)
    num_parameters = temp_model.num_non_embedding_parameters()
    
    if config.learning_rate is None:
        actual_lr = language_model_training.get_auto_learning_rate(num_parameters)
        config = replace(config, learning_rate=actual_lr)
    else:
        actual_lr = config.learning_rate
    
    del temp_model  # Free memory
    
    # Create run name
    nonlinearity = config.model_config.nonlinearity
    lr_str = f"{actual_lr:.4f}".rstrip('0').rstrip('.')
    cf_str = f"_cf{chinchilla_factor}" if chinchilla_factor != 1.0 else ""
    wd_str = f"_wd{weight_decay}" if weight_decay is not None and weight_decay != 0.0 else ""
    init_str = "_depthscale" if use_depth_scaling else ""
    
    run_name = f"{config.name}_lr{lr_str}{cf_str}{wd_str}_{nonlinearity}{init_str}"
    
    # Calculate training details
    num_tokens_per_step = config.batch_size * config.sequence_length
    chinchilla_optimal_steps = int(
        20 * num_parameters * config.chinchilla_factor / num_tokens_per_step
    )
    total_tokens = chinchilla_optimal_steps * num_tokens_per_step
    
    # Print configuration
    print("\n" + "=" * 80)
    print(f"{'DRY RUN' if dry_run else 'TRAINING'}: Custom Experiment")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Parameters (non-embedding): {num_parameters:,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Learning rate: {actual_lr}")
    print(f"  Weight decay: {weight_decay if weight_decay is not None else 0.0}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Chinchilla factor: {chinchilla_factor}")
    print(f"  Use proper init: {use_proper_init}")
    print(f"  Use depth scaling: {use_depth_scaling}")
    print(f"  Nonlinearity: {nonlinearity}")
    print(f"  Adam epsilon: {config.adam_eps}")
    print(f"  Adam betas: {config.adam_betas}")
    print(f"  Training steps: {chinchilla_optimal_steps:,} (Chinchilla-optimal × {chinchilla_factor})")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print(f"  Run name: {run_name}")
    print(f"  Neptune tags: {neptune_tags}")
    print("=" * 80)
    
    if dry_run:
        print("\n[DRY RUN] Skipping actual training")
        print("To run for real, remove the --dry_run flag")
        return
    
    # Create description
    description = (
        f"Training {model_name} with lr={actual_lr}, wd={config.weight_decay}, "
        f"warmup={config.warmup_steps}, proper_init={use_proper_init}, "
        f"depth_scaling={use_depth_scaling}"
    )

    # Run training
    language_model_training.run(
        config=config,
        run_name=run_name,
        description=description,
        use_neptune=True,
        neptune_tags=neptune_tags,
    )
    
    print(f"\nTraining complete: {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a custom training experiment with specific hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Your specific experiment
  python run_custom_experiment.py \\
      --model_size 117m \\
      --neptune_tags custom_experiment \\
      --learning_rate 0.003 \\
      --weight_decay 0.1 \\
      --warmup_steps 500 \\
      --chinchilla_factor 1.0 \\
      --use_proper_init \\
      --no_depth_scaling

  # Dry run to preview
  python run_custom_experiment.py \\
      --model_size 117m \\
      --neptune_tags test \\
      --learning_rate 0.003 \\
      --weight_decay 0.1 \\
      --warmup_steps 500 \\
      --dry_run

  # Quick test with reduced training
  python run_custom_experiment.py \\
      --model_size 74m \\
      --neptune_tags quick_test \\
      --learning_rate 0.002 \\
      --chinchilla_factor 0.5 \\
      --dry_run
        """
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        help="Model size (e.g., 74m, 117m, 251m)",
    )
    parser.add_argument(
        "--neptune_tags",
        type=str,
        nargs="+",
        required=True,
        help="Tags for Neptune logging",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (if not specified, auto-select based on model size)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay for AdamW optimizer (default: 0.0)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Number of warmup steps (default: 100)",
    )
    parser.add_argument(
        "--chinchilla_factor",
        type=float,
        default=1.0,
        help="Multiplier for Chinchilla-optimal training steps (default: 1.0)",
    )
    parser.add_argument(
        "--use_proper_init",
        action="store_true",
        default=False,
        help="Use activation-aware initialization (SiLU-aware)",
    )
    parser.add_argument(
        "--no_proper_init",
        dest="use_proper_init",
        action="store_false",
        help="Disable activation-aware initialization",
    )
    parser.add_argument(
        "--use_depth_scaling",
        action="store_true",
        default=False,
        help="Use depth-scaled initialization for output projections",
    )
    parser.add_argument(
        "--no_depth_scaling",
        dest="use_depth_scaling",
        action="store_false",
        help="Disable depth-scaled initialization (default)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (if not specified, auto-select based on model size)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print configuration without actually running training",
    )
    
    # Set defaults for proper_init
    parser.set_defaults(use_proper_init=True)
    
    args = parser.parse_args()
    
    run_custom_experiment(
        model_size=args.model_size,
        neptune_tags=args.neptune_tags,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        chinchilla_factor=args.chinchilla_factor,
        use_proper_init=args.use_proper_init,
        use_depth_scaling=args.use_depth_scaling,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

