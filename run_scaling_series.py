"""Entry point for running a series of LLM training experiments.

- Manages a single scaling experiment with multiple LLM training runs
  - at different scale
  - at same scale with different hyper parameters, like
    - learning rate, because stability
    - x * Chinchilla data scaling, because data efficiency
    These hyper parameters are intended to be "within" the scale,
    i.e. we want to take the best hyper parameter combination for each scale.
- Determine compute pareto frontier for a scaling experiment.
- Manage multiple GPUs
  - scheduling multiple runs in parallel
  - multi-GPU training runs [later]
- Notify per email when an experiment series or single run is done or failed.

"""

import language_model_training
from dataclasses import replace
import argparse


def config_variants(
    config: language_model_training.LanguageModelTrainingConfig,
) -> list[language_model_training.LanguageModelTrainingConfig]:
    """Generate variants of a config."""
    variants = [config]

    eps_variants = []
    for config in variants:
        for eps in [1e-7, 1e-6, 1e-8]:
            eps_variants.append(
                replace(
                    config,
                    adam_eps=eps,
                    name=config.name + f"_eps{eps}",
                )
            )
    variants = eps_variants

    betas_variants = []
    for config in variants:
        for betas in [(0.9, 0.99), (0.9, 0.95), (0.9, 0.995)]:
            betas_variants.append(
                replace(
                    config,
                    adam_betas=betas,
                    name=config.name + f"_betas{betas[1]}",
                )
            )
    variants = betas_variants

    bs_variants = []
    for config in variants:
        for bs in [128, 256, 512]:
            bs_variants.append(
                replace(
                    config,
                    batch_size=bs,
                    name=config.name + f"_bs{bs}",
                )
            )
    variants = bs_variants

    lr_variants = []
    for config in variants:
        for lr in [0.001, 0.0005, 0.0002]:
            lr_variants.append(
                replace(
                    config,
                    learning_rate=lr,
                    name=config.name + f"_lr{lr}",
                )
            )
    variants = lr_variants

    warmup_variants = []
    for config in variants:
        for warmup_steps in [100, 500]:
            warmup_variants.append(
                replace(
                    config,
                    warmup_steps=warmup_steps,
                    name=config.name + f"_warmup{warmup_steps}",
                )
            )
    variants = warmup_variants
    print(f"Generated {len(variants)} variants")
    return variants


def main(split: int, num_splits: int, neptune_tags: list[str]):
    configs = [
        # "chinchilla-44m",
        "chinchilla-74m",
        # "chinchilla-90m",
        # "chinchilla-106m",
        "chinchilla-140m",
        # "chinchilla-163m",
        "chinchilla-251m",
        # "chinchilla-306m",
        # "chinchilla-425m",
        "chinchilla-489m",
        # "chinchilla-632m",
        # "chinchilla-816m",
        # "chinchilla-1266m",
        # "chinchilla-1593m",
        # "chinchilla-2298m",
        # "chinchilla-4516m",
        # "chinchilla-9293m",
    ]

    # create all variants
    all_configs = []
    for config_str in configs:
        cfg = language_model_training.get_model_config(config_str)
        all_configs.extend(config_variants(cfg))

    print(f"Sweeping a total of {len(all_configs)} configs")

    # split configs
    all_configs = all_configs[split::num_splits]
    print(f"Running {len(all_configs)} configs on this GPU.")

    for cfg in all_configs:
        print(f"Running {cfg.name}")
        language_model_training.run(
            config=cfg,
            run_name=cfg.name,
            description=cfg.name,
            use_neptune=True,
            gpu_id=split,
            neptune_tags=neptune_tags,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Arguments that help split experiments over GPUs
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--neptune_tags", type=str, nargs="+", default=[])
    args = parser.parse_args()

    main(args.split, args.num_splits, args.neptune_tags)
