"""Train a series of Chinchilla models sequentially on a single GPU.

This script trains the following Chinchilla models:
- 74M, 106M, 163M, 196M, 251M, 306M, 425M, 489M, 632M, 816M

Features:
- Sequential training (one model at a time)
- Automatic learning rate selection (handled by language_model_training.py)
- Batch size selection based on model size
- Neptune logging with custom tags
- Optional filtering by model size
"""

import argparse
from dataclasses import replace

import language_model_training
import transformer
import model_configs.chinchilla  # noqa: F401 - needed to register configs


def get_batch_size(model_name: str) -> int:
    """Select batch size based on model size.

    Args:
        model_name: Name of the model (e.g., "chinchilla-74m")

    Returns:
        Appropriate batch size for the model
    """
    # Extract size in millions from model name
    size_str = model_name.replace("chinchilla-", "").replace("m", "")
    size_m = int(size_str)

    # Use smaller batch size for models <= 400M params
    if size_m <= 400:
        return 192
    else:
        return 256


def get_all_chinchilla_models() -> list[str]:
    """Get all registered Chinchilla models from the transformer config registry.

    Returns:
        List of model names sorted by size (e.g., ["chinchilla-44m", "chinchilla-74m", ...])
    """
    all_configs = transformer.transformer_config_registry.list_configs()
    chinchilla_models = [name for name in all_configs if name.startswith("chinchilla-")]

    # Sort by parameter size (extract number from name like "chinchilla-74m")
    def get_size(name: str) -> int:
        size_str = name.replace("chinchilla-", "").replace("m", "")
        return int(size_str)

    return sorted(chinchilla_models, key=get_size)


def main(
    neptune_tags: list[str],
    model_sizes: list[str] | None = None,
    dry_run: bool = False,
    training_steps: int | None = None,
    warmup_steps: int | None = None,
    depth_scaling_variants: bool = False,
    chinchilla_factor: float | None = None,
):
    """Train Chinchilla models sequentially.

    Args:
        neptune_tags: List of tags to apply to all Neptune runs
        model_sizes: Optional list of model sizes to train (e.g., ["74m", "106m"]).
                     If None, trains all default models.
        dry_run: If True, only print configuration without running training
        training_steps: Optional number of training steps to override Chinchilla-optimal.
                        Useful for quick verification runs.
        warmup_steps: Optional number of warmup steps to override default (100).
                      Default is 100 steps if not specified.
        depth_scaling_variants: If True, trains each model twice - once with depth_scaling=True
                                and once with depth_scaling=False.
        chinchilla_factor: Optional multiplier for Chinchilla-optimal training steps.
                           Default is 1.0 (20 tokens per parameter).
                           Use 2.0 for 40 tokens per parameter, 0.5 for 10 tokens per parameter, etc.
    """
    # Get all available Chinchilla models from registry
    all_available_models = get_all_chinchilla_models()

    # Default models to train if none specified
    default_model_names = [
        # "chinchilla-74m",
        # "chinchilla-106m",
        # "chinchilla-163m",
        "chinchilla-117m",
        # "chinchilla-251m",
        # "chinchilla-306m",
        # "chinchilla-425m",
        # "chinchilla-489m",
        # "chinchilla-632m",
        # "chinchilla-816m",
    ]

    # Filter models if specific sizes requested
    if model_sizes:
        model_names = [f"chinchilla-{size}" for size in model_sizes]
        # Validate that requested models exist in registry
        invalid_models = [m for m in model_names if m not in all_available_models]
        if invalid_models:
            available_sizes = [m.replace("chinchilla-", "") for m in all_available_models]
            raise ValueError(
                f"Invalid model sizes: {[m.replace('chinchilla-', '') for m in invalid_models]}. "
                f"Available in registry: {available_sizes}"
            )
    else:
        model_names = default_model_names

    # Create list of (model_name, use_depth_scaling) tuples
    if depth_scaling_variants:
        # Train each model twice: once with depth_scaling=False, once with depth_scaling=True
        training_configs = []
        for model_name in model_names:
            training_configs.append((model_name, False))
            training_configs.append((model_name, True))
    else:
        # Train each model once with default depth_scaling=False
        training_configs = [(model_name, False) for model_name in model_names]

    mode_str = "DRY RUN" if dry_run else "TRAINING"
    print(f"\n{'=' * 80}")
    print(f"{mode_str}: {len(training_configs)} training runs")
    print(f"{'=' * 80}")
    print(f"Models: {', '.join([m.replace('chinchilla-', '') for m in model_names])}")
    if depth_scaling_variants:
        print(f"Depth scaling variants: Training each model with depth_scaling=False and depth_scaling=True")
    print(f"Neptune tags: {neptune_tags}")
    print(f"Dry run mode: {dry_run}")
    if training_steps is not None:
        print(f"Training steps (manual override): {training_steps:,}")
    elif chinchilla_factor is not None:
        print(f"Training steps: Chinchilla-optimal × {chinchilla_factor} ({20 * chinchilla_factor:.0f} tokens per parameter)")
    else:
        print(f"Training steps: Chinchilla-optimal (20 tokens per parameter)")
    if warmup_steps is not None:
        print(f"Warmup steps (manual override): {warmup_steps:,}")
    else:
        print(f"Warmup steps: 100 (default)")
    print("-" * 80)

    for i, (model_name, use_depth_scaling) in enumerate(training_configs, 1):
        variant_str = "depth_scaling=True" if use_depth_scaling else "depth_scaling=False"
        print(f"\n[{i}/{len(training_configs)}] Starting training: {model_name} ({variant_str})")
        print("=" * 80)

        # Get base config for this model
        config = language_model_training.get_model_config(model_name)

        # Update model config to use depth scaling if requested
        if use_depth_scaling:
            model_config = replace(config.model_config, use_depth_scaling=True)
            config = replace(config, model_config=model_config)

        # Set batch size based on model size
        batch_size = get_batch_size(model_name)
        config = replace(config, batch_size=batch_size)

        # Override training steps if specified
        if training_steps is not None:
            import training_basics
            config = replace(
                config,
                training_config=replace(
                    config.training_config,
                    training_steps_per_epoch=training_steps,
                ),
            )

        # Override warmup steps if specified
        if warmup_steps is not None:
            config = replace(config, warmup_steps=warmup_steps)

        # Override chinchilla_factor if specified
        if chinchilla_factor is not None:
            config = replace(config, chinchilla_factor=chinchilla_factor)

        # Get nonlinearity from model config
        nonlinearity = config.model_config.nonlinearity

        # Pre-compute the learning rate that will be auto-selected
        # We need to create a temporary model to get parameter count
        import transformer
        import prng
        with prng.PRNG(config.seed + 123123):
            temp_model = transformer.TransformerModel(config.vocab_size, config.model_config)
        num_parameters = temp_model.num_non_embedding_parameters()
        learning_rate = language_model_training.get_auto_learning_rate(num_parameters)
        del temp_model  # Free memory

        # Create run name with learning rate, nonlinearity, and depth scaling
        # Format: c117_lr0.002_swish_depthscale or c117_lr0.002_swish
        lr_str = f"{learning_rate:.4f}".rstrip('0').rstrip('.')
        if use_depth_scaling:
            run_name = f"{config.name}_lr{lr_str}_{nonlinearity}_cf{chinchilla_factor}_depthscale"
        else:
            run_name = f"{config.name}_lr{lr_str}_cf{chinchilla_factor}_{nonlinearity}"

        print(f"Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Parameters (non-embedding): {num_parameters:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {config.sequence_length}")
        print(f"  Learning rate: {learning_rate} (auto-selected)")
        print(f"  Nonlinearity: {nonlinearity}")
        print(f"  Use depth scaling: {use_depth_scaling}")
        print(f"  Chinchilla factor: {config.chinchilla_factor}")
        print(f"  Adam epsilon: {config.adam_eps}")
        print(f"  Adam betas: {config.adam_betas}")
        if warmup_steps is not None:
            print(f"  Warmup steps: {config.warmup_steps} (manual override)")
        else:
            print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Run name: {run_name}")

        # Create description with actual learning rate and depth scaling info
        depth_scale_str = "with depth scaling" if use_depth_scaling else "without depth scaling"
        description = (
            f"Training model {model_name} {depth_scale_str}, "
            f"learning rate {learning_rate}, {nonlinearity}"
        )
        print(f"  Description: {description}")

        # Calculate training details
        num_tokens_per_step = config.batch_size * config.sequence_length
        if config.training_config.training_steps_per_epoch is None:
            chinchilla_optimal_steps = int(
                20 * num_parameters * config.chinchilla_factor / num_tokens_per_step
            )
            actual_steps = chinchilla_optimal_steps
            print(f"  Training steps: {actual_steps:,} (Chinchilla-optimal)")
        else:
            actual_steps = config.training_config.training_steps_per_epoch
            if training_steps is not None:
                print(f"  Training steps: {actual_steps:,} (manual override)")
            else:
                print(f"  Training steps: {actual_steps:,}")

        total_tokens = actual_steps * num_tokens_per_step
        print(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")

        print("-" * 80)

        if dry_run:
            print(f"[DRY RUN] Skipping actual training for {model_name}")
        else:
            # Run training
            language_model_training.run(
                config=config,
                run_name=run_name,
                description=description,
                use_neptune=True,
                neptune_tags=neptune_tags,
            )
            print(f"\n[{i}/{len(training_configs)}] Completed: {model_name} ({variant_str})")

        print("=" * 80)
    
    print("\n" + "=" * 80)
    if dry_run:
        print("DRY RUN COMPLETE - No models were actually trained")
        print(f"To run for real, remove the --dry_run flag")
    else:
        print("All models trained successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a series of Chinchilla models sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what will be trained
  python train_chinchilla_series.py --neptune_tags baseline_experiment v1 --dry_run

  # Train all default models
  python train_chinchilla_series.py --neptune_tags baseline_experiment v1

  # Train specific model sizes
  python train_chinchilla_series.py --neptune_tags test_run --model_sizes 74m 106m

  # Train a single model
  python train_chinchilla_series.py --neptune_tags quick_test --model_sizes 74m

  # Dry run for specific models
  python train_chinchilla_series.py --neptune_tags test --model_sizes 74m 816m --dry_run

  # Quick verification run with limited steps
  python train_chinchilla_series.py --neptune_tags verify --model_sizes 74m --training_steps 100

  # Dry run with custom steps to preview
  python train_chinchilla_series.py --neptune_tags test --model_sizes 74m --training_steps 1000 --dry_run

  # Custom warmup steps for experimentation
  python train_chinchilla_series.py --neptune_tags experiment --model_sizes 74m --warmup_steps 500

  # Quick run with both custom training and warmup steps
  python train_chinchilla_series.py --neptune_tags quick_test --model_sizes 74m --training_steps 100 --warmup_steps 10

  # Compare depth scaling variants for a specific model
  python train_chinchilla_series.py --neptune_tags depth_scaling_experiment --model_sizes 117m --depth_scaling_variants

  # Dry run to preview depth scaling variants
  python train_chinchilla_series.py --neptune_tags test --model_sizes 117m --depth_scaling_variants --dry_run

  # Train with 2x Chinchilla-optimal data (40 tokens per parameter)
  python train_chinchilla_series.py --neptune_tags extended_training --model_sizes 117m --chinchilla_factor 2.0

  # Train with 0.5x Chinchilla-optimal data (10 tokens per parameter) for quick testing
  python train_chinchilla_series.py --neptune_tags quick_test --model_sizes 74m --chinchilla_factor 0.5
        """
    )
    parser.add_argument(
        "--neptune_tags",
        type=str,
        nargs="+",
        required=True,
        help="Tags to apply to all Neptune runs (e.g., experiment_name version_1)",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        nargs="+",
        default=None,
        help="Specific model sizes to train (e.g., 74m 106m 163m). If not provided, trains all models.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=None,
        help="Number of training steps (overrides Chinchilla-optimal). Useful for quick verification runs.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Number of warmup steps (overrides default of 100). Useful for experimentation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print configuration without actually running training",
    )
    parser.add_argument(
        "--depth_scaling_variants",
        action="store_true",
        default=False,
        help="Train each model twice: once with depth_scaling=False and once with depth_scaling=True",
    )
    parser.add_argument(
        "--chinchilla_factor",
        type=float,
        default=None,
        help="Multiplier for Chinchilla-optimal training steps (default: 1.0 = 20 tokens per parameter). "
             "Use 2.0 for 40 tokens/param, 0.5 for 10 tokens/param, etc.",
    )

    args = parser.parse_args()

    main(
        neptune_tags=args.neptune_tags,
        model_sizes=args.model_sizes,
        dry_run=args.dry_run,
        training_steps=args.training_steps,
        warmup_steps=args.warmup_steps,
        depth_scaling_variants=args.depth_scaling_variants,
        chinchilla_factor=args.chinchilla_factor,
    )

