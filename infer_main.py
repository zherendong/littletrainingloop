import argparse
import checkpointing
import lm_eval_wrapper
from pathlib import Path
import torch

device = "cuda"
torch.set_default_device(device)


def infer(checkpoint_path: str, prompt: str, generate_until_max_length: int = 256):
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = checkpointing.load_model_from_training_checkpoint(
        Path(checkpoint_path),
        device=device,
    )
    print(f"Loaded checkpoint from {checkpoint_path}.")
    model = ckpt.model
    config = ckpt.config

    wrapper = lm_eval_wrapper.LittleTrainingLoopWrapper(
        model=model,
        config=config,
        device=device,
        batch_size=1,
        generate_until_max_length=generate_until_max_length,
    )
    print(wrapper.infer(prompt, until=[]))


if __name__ == "__main__":

    """Example usage:

    python infer_main.py --checkpoint_path /path/to/checkpoint --prompt "Hello world"
    """

    # command line args, including name
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    assert args.checkpoint_path is not None, "Please provide a checkpoint path"
    assert args.prompt is not None, "Please provide a prompt"
    infer(args.checkpoint_path, args.prompt)
