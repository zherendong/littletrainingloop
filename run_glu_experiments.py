#!/usr/bin/env python3
"""
GLU Experiment Runner - Local Sequential Execution

This script runs a series of experiments to compare GLU variants against baseline FFN.
Unlike run_scaling_series.py, this is designed for local execution on a single GPU.

The script tests:
1. Baseline (glu=False, nonlinearity="gelu")
2. GeGLU (glu=True, nonlinearity="gelu")
3. SwiGLU (glu=True, nonlinearity="swish")

All experiments use inner_size_multiple_of=64 for fair parameter matching.

Usage:
    # Run all experiments with Chinchilla-optimal steps (recommended)
    python run_glu_experiments.py --model_size chinchilla-44m --no_neptune

    # Quick test run (manual step override)
    python run_glu_experiments.py --model_size chinchilla-44m --steps 500 --no_neptune

    # Run with specific LR sweep
    python run_glu_experiments.py --model_size chinchilla-44m --lr_sweep 0.0015 0.001 0.0007

    # Run only specific variants (baseline + SwiGLU)
    python run_glu_experiments.py --model_size chinchilla-44m --variants baseline swiglu --no_neptune
"""

import argparse
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import language_model_training
import transformer


def create_glu_variants(
    base_model_config_name: str,
    learning_rates: list[float],
    warmup_steps: int,
    inner_size_multiple_of: int = 64,
    include_variants: list[str] | None = None,
) -> list[tuple[str, language_model_training.LanguageModelTrainingConfig]]:
    """
    Create experiment configs for baseline and GLU variants.
    
    Args:
        base_model_config_name: e.g., "chinchilla-44m"
        learning_rates: List of learning rates to sweep
        warmup_steps: Number of warmup steps
        inner_size_multiple_of: Rounding factor for MLP hidden size (64 recommended for GLU)
        include_variants: List of variants to include (None = all). Options: baseline, geglu, swiglu
    
    Returns:
        List of (variant_name, config) tuples
    """
    base_model_config = transformer.transformer_config_registry.get(base_model_config_name)
    
    # Define all possible variants
    all_variants = {
        "baseline": {
            "glu": False,
            "nonlinearity": "gelu",
            "suffix": "baseline",
        },
        "geglu": {
            "glu": True,
            "nonlinearity": "gelu",
            "suffix": "geglu",
        },
        "swiglu": {
            "glu": True,
            "nonlinearity": "swish",
            "suffix": "swiglu",
        },
    }
    
    # Filter variants if requested
    if include_variants:
        variants_to_run = {k: v for k, v in all_variants.items() if k in include_variants}
    else:
        variants_to_run = all_variants
    
    configs = []
    
    for lr in learning_rates:
        for variant_key, variant_spec in variants_to_run.items():
            # Create modified model config
            model_config = replace(
                base_model_config,
                glu=variant_spec["glu"],
                nonlinearity=variant_spec["nonlinearity"],
                inner_size_multiple_of=inner_size_multiple_of,
            )
            
            # Create training config
            base_name = base_model_config_name.replace("chinchilla-", "c")
            lr_str = f"lr{lr:.0e}".replace("e-0", "e-")  # e.g., "lr1.5e-3"
            variant_name = f"{base_name}_{variant_spec['suffix']}_{lr_str}_w{warmup_steps}"
            
            training_config = language_model_training.LanguageModelTrainingConfig(
                name=variant_name,
                vocab_size=100277,
                warmup_steps=warmup_steps,
                learning_rate=lr,
                batch_size=192,
                sequence_length=512,
                shuffle_buffer_size=100,
                adam_eps=1e-7,
                adam_betas=(0.9, 0.995),
                training_config=language_model_training.TrainingConfig(
                    num_epochs=1,
                    training_steps_per_epoch=None,  # Will be overridden by command line
                    seed=42,
                ),
                eval_config=language_model_training.EvalConfig(
                    every_n_steps=100,
                    steps=5,
                    batch_size=256,
                    sequence_length=512,
                ),
                model_config=model_config,
            )
            
            configs.append((variant_name, training_config))
    
    return configs


def run_experiment_local(
    config: language_model_training.LanguageModelTrainingConfig,
    use_neptune: bool,
    max_steps: int | None = None,
    description: str | None = None,
    gpu_id: int | None = None,
) -> bool:
    """
    Run a single experiment locally using language_model_training.py.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Starting experiment: {config.name}")
    print(f"  Model config: glu={config.model_config.glu}, "
          f"nonlinearity={config.model_config.nonlinearity}, "
          f"inner_size_multiple_of={config.model_config.inner_size_multiple_of}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    if max_steps:
        print(f"  Max training steps: {max_steps}")
    print(f"  Neptune logging: {use_neptune}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "language_model_training.py",
        "--model_config", "chinchilla-44m",  # We'll override with our config
        "--name", config.name,
    ]
    
    if not use_neptune:
        cmd.append("--no_neptune")
    
    if description:
        cmd.extend(["--description", description])
    
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    
    # Note: We can't easily pass the full config via command line
    # Instead, we'll need to modify this to call the run() function directly
    # For now, let's use a simpler approach with direct function call
    
    try:
        # Import and run directly instead of subprocess
        # This allows us to pass the custom config
        from language_model_training import run
        
        # Override training steps if specified
        # If max_steps is None, training_steps_per_epoch stays None and Chinchilla-optimal is used
        if max_steps is not None:
            config = replace(
                config,
                training_config=replace(
                    config.training_config,
                    training_steps_per_epoch=max_steps,
                ),
            )
        
        run(
            config=config,
            use_neptune=use_neptune,
            description=description,
            run_name=config.name,
            gpu_id=gpu_id,
            neptune_tags=["glu_experiment"],
            use_metrics_logger=not use_neptune,  # Use metrics logger when Neptune is disabled
        )
        
        print(f"\n✓ Experiment {config.name} completed successfully\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Experiment {config.name} failed with error: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run GLU experiments locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        default="chinchilla-44m",
        help="Base model size (e.g., chinchilla-44m, chinchilla-106m)",
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps per experiment (default: None = Chinchilla-optimal, ~20 tokens per parameter)",
    )
    
    parser.add_argument(
        "--lr_sweep",
        type=float,
        nargs="+",
        default=None,
        help="Learning rates to sweep (default: auto-select based on model size)",
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500, recommended for GLU)",
    )
    
    parser.add_argument(
        "--inner_size_multiple_of",
        type=int,
        default=64,
        help="Rounding factor for MLP hidden size (default: 64 for fair GLU comparison)",
    )
    
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=["baseline", "geglu", "swiglu"],
        default=None,
        help="Which variants to run (default: all)",
    )
    
    parser.add_argument(
        "--no_neptune",
        action="store_true",
        help="Disable Neptune logging",
    )
    
    parser.add_argument(
        "--description",
        "-d",
        type=str,
        default="GLU variant comparison experiment",
        help="Experiment description for Neptune",
    )
    
    parser.add_argument(
        "--gpu_id",
        "-g",
        type=int,
        default=None,
        help="GPU ID to use (default: auto-select)",
    )
    
    args = parser.parse_args()
    
    # Determine learning rates
    if args.lr_sweep:
        learning_rates = args.lr_sweep
    else:
        # Auto-select based on model size (similar to run_scaling_series.py)
        try:
            size_str = args.model_size.replace("chinchilla-", "").replace("m", "")
            model_size_m = int(size_str)
            
            if model_size_m <= 100:
                learning_rates = [0.0015, 0.001, 0.0007]  # Sweep for small models
            elif model_size_m <= 200:
                learning_rates = [0.0015]
            elif model_size_m <= 300:
                learning_rates = [0.001]
            else:
                learning_rates = [0.0007, 0.001]
        except ValueError:
            print(f"Warning: Could not parse model size from {args.model_size}, using default LR")
            learning_rates = [0.0015]
    
    print(f"\n{'='*80}")
    print(f"GLU Experiment Configuration")
    print(f"{'='*80}")
    print(f"Model size: {args.model_size}")
    if args.steps is None:
        print(f"Training steps per experiment: Chinchilla-optimal (auto, ~20 tokens per parameter)")
    else:
        print(f"Training steps per experiment: {args.steps} (manual override)")
    print(f"Learning rates: {learning_rates}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"inner_size_multiple_of: {args.inner_size_multiple_of}")
    print(f"Variants: {args.variants if args.variants else 'all (baseline, geglu, swiglu)'}")
    print(f"Neptune logging: {not args.no_neptune}")
    print(f"{'='*80}\n")
    
    # Create experiment configs
    configs = create_glu_variants(
        base_model_config_name=args.model_size,
        learning_rates=learning_rates,
        warmup_steps=args.warmup_steps,
        inner_size_multiple_of=args.inner_size_multiple_of,
        include_variants=args.variants,
    )
    
    print(f"Generated {len(configs)} experiment configurations\n")
    
    # Run experiments sequentially
    results = []
    start_time = time.time()
    
    for i, (variant_name, config) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running: {variant_name}")
        
        success = run_experiment_local(
            config=config,
            use_neptune=not args.no_neptune,
            max_steps=args.steps,
            description=args.description,
            gpu_id=args.gpu_id,
        )
        
        results.append((variant_name, success))
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Experiment Series Complete")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nResults:")
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\nSummary: {successful}/{len(results)} experiments completed successfully")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

