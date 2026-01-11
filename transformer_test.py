"""
Test script for the transformer.py module.

Mostly smoke tests
"""

import torch
from torch.utils import flop_counter
import transformer
import model_configs.mini_transformers  # noqa: F401
from transformer import (
    MLP,
    SelfAttention,
    TransformerBlock,
    TransformerModel,
    TransformerConfig,
)

if torch.cuda.is_available():
    torch.set_default_device("cuda")


def test_mlp():
    """Smoke test for the MLP class"""
    mlp = MLP(
        dtype=torch.bfloat16,
        input_size=128,
        output_size=256,
    )
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    y = mlp(x)
    assert y.shape == (3, 256)
    assert not torch.isnan(y).any()


def test_self_attention():
    """Smoke test for the SelfAttention class"""
    attention = SelfAttention(
        input_size=128,
        num_heads_q=8,
        num_heads_kv=2,
        head_dim=32,
        use_flash_attention=False,
        dtype=torch.bfloat16,
        max_seq_len=512,
    )
    x = torch.randn(3, 10, 128, dtype=torch.bfloat16)
    y = attention(x)
    assert y.shape == (3, 10, 128)
    assert not torch.isnan(y).any()


def test_transformer_block():
    """Smoke test for the TransformerBlock class"""

    def mlp_factory():
        return MLP(
            dtype=torch.bfloat16,
            input_size=128,
            output_size=128,
            inner_size=512,
        )

    def attention_factory():
        return SelfAttention(
            input_size=128,
            num_heads_q=8,
            num_heads_kv=2,
            head_dim=32,
            use_flash_attention=False,
            dtype=torch.bfloat16,
        )

    config = TransformerConfig(
        num_layers=2,
        num_heads=8,
        num_heads_kv=2,
        head_dim=32,
        mlp_inner_size=512,
        embedding_size=128,
    )
    block = TransformerBlock(config=config, block_idx=0, params_dtype=torch.bfloat16)
    x = torch.randn(3, 10, 128, dtype=torch.bfloat16)
    y = block(x)
    assert y.shape == (3, 10, 128)
    assert not torch.isnan(y).any()


def test_transformer_model():
    """Smoke test for the TransformerModel class"""
    config = TransformerConfig(
        num_layers=2,
        num_heads=8,
        num_heads_kv=2,
        head_dim=32,
        mlp_inner_size=512,
        embedding_size=256,
    )
    vocab_size = 1024
    model = TransformerModel(vocab_size=vocab_size, config=config)
    x = torch.randint(0, vocab_size, (3, 10), dtype=torch.int32)
    flops = flop_counter.FlopCounterMode(depth=4)
    with flops:
        y = model(x)
    assert y.shape == (3, 10, vocab_size)
    assert not torch.isnan(y).any()
    assert flops.get_total_flops() in [
        100584480,
    ]  # may have different number of flops with flash attention
    assert sum(p.numel() for p in model.parameters()) in [
        1934592,
    ]  # may have different number of parameters with flash attention

    # now count with backward
    flops = flop_counter.FlopCounterMode(depth=4)
    with flops:
        y = model(x)
        loss = y.sum()
        loss.backward()
    assert flops.get_total_flops() in [368914560]


def test_mini_transformer_smoketest():
    """Smoke test for the mini transformer"""
    config = transformer.transformer_config_registry.get("mini-transformer-1")
    vocab_size = 1024
    model = TransformerModel(vocab_size=vocab_size, config=config)
    x = torch.randint(0, vocab_size, (3, 10), dtype=torch.int32)
    y = model(x)
    assert y.shape == (3, 10, vocab_size)
    assert not torch.isnan(y).any()
