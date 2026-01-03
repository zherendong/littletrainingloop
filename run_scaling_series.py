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
import math

import subprocess


def get_chinchilla_size(name: str) -> int:
    return int(name.split("_")[0].replace("c", "").replace("m", ""))


def config_variants(
    config: language_model_training.LanguageModelTrainingConfig,
) -> list[language_model_training.LanguageModelTrainingConfig]:
    """Generate variants of a config."""
    variants = [config]

    lr_variants = []
    for config in variants:
        # get chinchilla num params from config.name
        try:
            chinchilla_size = get_chinchilla_size(config.name)
        except IndexError:
            print(f"Could not parse chinchilla size from {config.name}")
            continue
        if chinchilla_size <= 50:
            lrs = [0.004]
        elif chinchilla_size <= 120:
            lrs = [0.003]
        elif chinchilla_size <= 180:
            lrs = [0.0025]
        elif chinchilla_size <= 300:
            lrs = [0.002]
        elif chinchilla_size <= 400:
            lrs = [0.0012]
        elif chinchilla_size <= 500:
            lrs = [0.0012]
        elif chinchilla_size <= 1000:
            lrs = [0.0007]
        else:
            lrs = [0.0007]
        for lr in lrs:
            lr_variants.append(
                replace(
                    config,
                    learning_rate=lr,
                    name=config.name + f"_lr{lr}",
                )
            )
    variants = lr_variants

    vanilla_variants = []
    for config in variants:
        # vanilla_variants.append(config)
        vanilla_variants.append(
            replace(
                config,
                model_config=replace(
                    config.model_config,
                    pre_projection_transform=None,
                ),
                name=config.name + "_vanilla",
            )
        )
    variants = vanilla_variants

    # spelling_bee_variants = []
    # for config in variants:
    #     # spelling_bee_variants.append(config)
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=True,
    #     #             spelling_bee_in_out_scale=1
    #     #             / math.sqrt(config.model_config.embedding_size),
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=True,
    #     #             apply_rotary=False,
    #     #             spelling_bee_in_out_scale=1
    #     #             / math.sqrt(config.model_config.embedding_size),
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_norotary",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=True,
    #     #             separate_token_embedding=False,
    #     #             spelling_bee_in_out_scale=1
    #     #             / math.sqrt(config.model_config.embedding_size),
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_notoken",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1
    #     #             / math.sqrt(config.model_config.embedding_size),
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_nocharnorm",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1,
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_nocharnorm_s",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_nocharnorm_mean",
    #     #     )
    #     # )

    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_type="dummy",
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_dummy",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_type="shuffled",
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_shuffled",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_type="full",
    #     #             spelling_bee_max_characters=1,
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_firstchar",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_bee_rotary_base=100,
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_rot100",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_bee_rotary_base=1000,
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_rot1000",
    #     #     )
    #     # )

    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_type="double",
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_doublei",
    #     #     )
    #     # )
    #     # spelling_bee_variants.append(
    #     #     replace(
    #     #         config,
    #     #         model_config=replace(
    #     #             config.model_config,
    #     #             spelling_bee=True,
    #     #             char_init_scale=1.0,
    #     #             char_embedding_norm=False,
    #     #             spelling_bee_in_out_scale=1.0,
    #     #             spelling_type="static_emb",
    #     #         ),
    #     #         name=config.name + f"_spellingbee_s_static",
    #     #     )
    #     # )

    spelling_bee_variants = []
    for config in variants:
        # spelling_bee_variants.append(config)
        # # spelling bee
        # spelling_bee_variants.append(
        #     replace(
        #         config,
        #         model_config=replace(
        #             config.model_config,
        #             spelling_bee=True,
        #             char_init_scale=1.0,
        #             char_embedding_norm=False,
        #             spelling_bee_in_out_scale=1.0,
        #             spelling_type="full",
        #         ),
        #         name=config.name + f"_spellingbee",
        #     )
        # )
        # static
        # spelling_bee_variants.append(
        #     replace(
        #         config,
        #         model_config=replace(
        #             config.model_config,
        #             spelling_bee=True,
        #             char_init_scale=1.0,
        #             char_embedding_norm=False,
        #             spelling_bee_in_out_scale=1.0,
        #             spelling_type="static_emb",
        #         ),
        #         name=config.name + f"_s_static",
        #     )
        # )
        # # no rotary
        # spelling_bee_variants.append(
        #     replace(
        #         config,
        #         model_config=replace(
        #             config.model_config,
        #             spelling_bee=True,
        #             char_init_scale=1.0,
        #             char_embedding_norm=False,
        #             spelling_bee_in_out_scale=1.0,
        #             spelling_type="full",
        #             apply_rotary=False,
        #         ),
        #         name=config.name + f"_s_norotary",
        #     )
        # )
        # no main token emb
        spelling_bee_variants.append(
            replace(
                config,
                model_config=replace(
                    config.model_config,
                    spelling_bee=True,
                    char_init_scale=1.0,
                    char_embedding_norm=False,
                    spelling_bee_in_out_scale=1.0,
                    spelling_type="full",
                    separate_token_embedding=False,
                ),
                name=config.name + f"_s_notoken",
            )
        )
    variants = spelling_bee_variants

    print(f"Generated {len(variants)} variants")
    return variants


def main(
    split: int,
    num_splits: int,
    neptune_tags: list[str],
    no_neptune: bool = False,
):
    configs = [  # extended group of models
        # "chinchilla-44m",
        # "chinchilla-74m",
        # "chinchilla-90m",
        # "chinchilla-106m",
        # "chinchilla-117m",
        # "chinchilla-140m",
        # "chinchilla-163m",
        # "chinchilla-196m",
        # "chinchilla-251m",
        # "chinchilla-306m",
        "chinchilla-425m",
        # "chinchilla-489m",
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
        cfg = language_model_training.get_model_config(
            config_str, checkpoint_path="../checkpoints"
        )
        cfg = replace(cfg, name=config_str.replace("chinchilla-", "c"))
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
            use_neptune=not no_neptune,
            neptune_tags=neptune_tags,
        )


# Define GPU machines (excluding lambdagh200_1 for development)
gpu_machines = [
    "lambdagh200",
    "lambdagh200_2",
    "lambdagh200_3",
    "lambdagh200_4",
    "lambdagh200_5",
    "lambdagh200_6",
    "lambdagh200_7",
    "lambdagh200_8",
]


def multi_gpu_main(
    neptune_tags: list[str],
    gpus_to_use: list[int] | None = None,
    no_neptune: bool = False,
):
    """Run multiple experiments in parallel on multiple GPUs.

    We have lambdagh200_2, lambdagh200_3, ..., lambdagh200_8 and we want to split the work over them.
    This is 7 GPUs. There is also lambdagh200_1, but it is reserved for development.

    We ssh into each of them with `ssh lambdagh200_X`, then attach to tmux session "train" and run this script with
    python run_scaling_series.py --split 0 --num_splits 7 --neptune_tags [...]
    python run_scaling_series.py --split 1 --num_splits 7 --neptune_tags [...]
    ...
    etc.
    """

    if gpus_to_use is not None:
        gpu_machines_this_run = [gpu_machines[i] for i in gpus_to_use]
    else:
        gpu_machines_this_run = gpu_machines

    num_splits = len(gpu_machines_this_run)

    for split_idx, machine in enumerate(gpu_machines_this_run):
        # Build the Python command
        python_cmd = f"python run_scaling_series.py --split {split_idx} --num_splits {num_splits} --neptune_tags {' '.join(neptune_tags)}"
        if no_neptune:
            python_cmd += " --no_neptune"

        # SSH into machine and send command to tmux session "train"
        # tmux send-keys sends the command and Enter to execute it
        ssh_cmd = [
            "ssh",
            machine,
            f"tmux send-keys -t train '{python_cmd}' Enter",
        ]

        print(f"Launching on {machine} (split {split_idx}/{num_splits})")

        # Execute the SSH command
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Command sent to {machine}:train")
        else:
            print(f"  ✗ Failed to send command to {machine}")
            print(f"    Error: {result.stderr}")

    print(f"\nAll {num_splits} jobs dispatched!")
    print("To monitor progress, SSH into each machine and attach to tmux:")
    print(f"  ssh {gpu_machines_this_run[0]} -t 'tmux attach -t train'")


def stop_all_jobs():
    """Stop all jobs on all machines."""
    for machine in gpu_machines:
        ssh_cmd = [
            "ssh",
            machine,
            "tmux send-keys -t train C-c C-c Enter",
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Stop command sent to {machine}:train")
        else:
            print(f"  ✗ Failed to send stop command to {machine}")


def check_jobs():
    """Check if jobs are running on all machines."""
    # the session is always there, but the process might have died
    for machine in gpu_machines:
        # check if there is a python process running in the tmux session
        if machine == gpu_machines[0]:
            # go direct, not via ssh
            ssh_cmd = ["pgrep", "-f", "python run_scaling_series.py"]
        else:
            ssh_cmd = [
                "ssh",
                machine,
                'pgrep -f "python run_scaling_series.py"',
            ]
        # print(f"Checking {machine}:train")
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        # print(f"{result.returncode=}, {result.stdout=}, {result.stderr=}")
        if result.returncode == 0:
            print(f"  ✓ {machine}:train is running")
        else:
            print(f"  ✗ {machine}:train is not running")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Arguments that help split experiments over GPUs
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--neptune_tags", type=str, nargs="+", default=[])
    parser.add_argument("--stop_all_jobs", action="store_true", default=False)
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--check_jobs", action="store_true", default=False)
    parser.add_argument("--gpus_to_use", type=int, nargs="+", default=None)
    parser.add_argument("--no_neptune", action="store_true", default=False)
    args = parser.parse_args()

    assert (
        args.no_neptune or args.stop_all_jobs or args.check_jobs or args.neptune_tags
    ), "Please provide neptune tags"

    if args.stop_all_jobs:
        stop_all_jobs()
    elif args.check_jobs:
        check_jobs()
    elif args.multi_gpu:
        multi_gpu_main(args.neptune_tags, args.gpus_to_use, args.no_neptune)
    else:
        main(args.split, args.num_splits, args.neptune_tags, args.no_neptune)
