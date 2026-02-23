"""
Test growing MLP in transformer context using existing infrastructure.
"""

import dataclasses

import sys
sys.path.insert(0, "../..")
import transformer
import language_model_training
import shakespeare_dataloader
from training_basics import TrainingConfig
from language_model_basics import LanguageModelTrainingConfig, EvalConfig


def register_tiny_shakespeare_configs():
    """Register small configs for TinyShakespeare testing."""
    
    @transformer.transformer_config_registry.register("tiny-shakespeare")
    def tiny_static():
        return transformer.TransformerConfig(
            num_layers=2,
            num_heads=4,
            num_heads_kv=2,
            head_dim=32,
            # mlp_inner_size=256,
            embedding_size=128,
            max_seq_len=256,
            glu=False,
            nonlinearity="swish",
            pairwise_cancelling_init=False,
            pre_projection_transform=None,
            zheren_init=True,
        )


def get_tiny_shakespeare_config(
    growing: bool, 
    base_model_config_str: str, 
    total_training_steps: int, 
    growing_at: tuple | None = None, 
    block_steps: tuple | None = None
) -> LanguageModelTrainingConfig:
    """Create config for TinyShakespeare experiments."""
    vocab_size = shakespeare_dataloader.get_vocab_size()
    model_config = transformer.transformer_config_registry.get(base_model_config_str)
    model_config_name = base_model_config_str

    if growing:
        model_config = dataclasses.replace(
            model_config,
            growing_mlp=True,
            growing_mlp_block_size=128,
            growing_mlp_initial_blocks=1,
            growing_mlp_output_scale_on_add=True,
            add_block_at_steps=growing_at[1:], 
            block_schedule_lengths=block_steps[1:],  # Each new block's LR schedule length
        )
        model_config_name += '-growing'

    training_config = LanguageModelTrainingConfig(
        name=model_config_name,
        vocab_size=vocab_size,
        learning_rate=1e-3,
        batch_size=64,
        sequence_length=128,
        dataset="tiny_shakespeare",
        training_config=TrainingConfig(
            num_epochs=1,
            training_steps_per_epoch=total_training_steps,
            train_metrics_every_n_steps=100,
            seed=42,
            checkpoint_path=None,
        ),
        eval_config=EvalConfig(
            every_n_steps=200,
            steps=5,
            full_eval_every_n_steps=None,
            batch_size=64,  # match training
            sequence_length=128,  # match training
        ),
        model_config=model_config,
    )

    return training_config

def run_test(
    growing: bool, 
    total_steps: int, 
    growing_at: list | None = None, 
    block_steps: list | None = None
): 
    """Test growing or static transformer on TinyShakespeare."""
    config = get_tiny_shakespeare_config(
        growing, 
        "tiny-shakespeare",
        total_steps,
        growing_at,
        block_steps
    )

    if growing:
        run_name = "tiny_shakespeare_growing"
        description = "Growing MLP on TinyShakespeare"
    else:
        run_name="tiny_shakespeare_static",
        description="Static MLP baseline on TinyShakespeare"

    language_model_training.run(
        config=config,
        run_name=run_name,
        description=description,
        use_neptune=False,
    )

def compute_equivalent_static_steps(
    static_params: int,
    growing_params: list[int],
    growing_steps: list[int],
) -> int:
    """Compute equivalent training steps for a static model to match growing model compute.
    
    Args:
        static_params: Number of parameters in the static model.
        growing_params: Number of parameters at each phase of the growing model.
            E.g., [1000, 2000, 3000] means model has 1000 params, then 2000, then 3000.
        growing_steps: Number of steps in each phase.
            E.g., [500, 500, 500] means 500 steps at each size.
    
    Returns:
        Number of steps the static model should train for to match total compute
        (measured as sum of params * steps across all phases).
    
    Example:
        Growing model: 1000 params for 500 steps, then 2000 params for 500 steps
        Total compute: 1000*500 + 2000*500 = 1,500,000 param-steps
        
        Static model with 2000 params:
        Equivalent steps: 1,500,000 / 2000 = 750 steps
        
        >>> compute_equivalent_static_steps(2000, [1000, 2000], [500, 500])
        750
    """
    assert len(growing_params) == len(growing_steps), (
        f"Length mismatch: {len(growing_params)} param phases vs {len(growing_steps)} step phases"
    )
    
    total_param_steps = sum(p * s for p, s in zip(growing_params, growing_steps))
    equivalent_steps = total_param_steps // static_params
    
    return equivalent_steps


if __name__ == "__main__":
    register_tiny_shakespeare_configs()

    block_size = 128
    blocks_at = [0, 1000, 2000, 2500]
    train_for = [3000, 2000, 1000, 500]
    total_growing_steps = 3000

    # approximate param matching
    # assumes in & out of mlps is the same 
    static_size = block_size * len(blocks_at) # static model equal to final grown model
    growing_params = [block_size] * len(blocks_at)
    static_training_steps = compute_equivalent_static_steps(static_size, growing_params, train_for)
    # very rough hack to account for embedding and attention params, assuming 25% of model size is fixed
    static_training_steps = int(static_training_steps*0.75 + total_growing_steps*0.25)

    print("=" * 50)
    print("Test: Static MLP on TinyShakespeare")
    print(f"Static model will train for {static_training_steps} steps")
    print("=" * 50)
    run_test(False, static_training_steps)

    print("=" * 50)
    print("Test: Growing MLP on TinyShakespeare")
    print(f"Growing model will train for {total_growing_steps} steps")
    print("=" * 50)
    run_test(True, total_growing_steps, blocks_at, train_for)