"""
Test script for the transformer.py module.

Mostly smoke tests
"""

import torch
from transformer import (
    MLP,
    SelfAttention,
    TransformerBlock,
    TransformerModel,
    TransformerConfig,
    attention_fn,
)


def test_mlp():
    """Smoke test for the MLP class"""
    mlp = MLP(input_size=128, output_size=256, inner_size=512)
    x = torch.randn(3, 128)
    y = mlp(x)
    assert y.shape == (3, 256)
    assert not torch.isnan(y).any()


def test_attention_fn():
    """Smoke test for the attention_fn function"""
    q = torch.randn(1, 3, 2, 4, 7)
    k = torch.randn(1, 3, 2, 7)
    v = torch.randn(1, 3, 2, 7)
    y = attention_fn(q, k, v)
    assert y.shape == (1, 3, 2, 4, 7)
    assert not torch.isnan(y).any()


def test_self_attention():
    """Smoke test for the SelfAttention class"""
    attention = SelfAttention(
        input_size=128, num_heads_q=8, num_heads_kv=2, head_dim=32
    )
    x = torch.randn(3, 10, 128)
    y = attention(x)
    assert y.shape == (3, 10, 128)
    assert not torch.isnan(y).any()


def test_transformer_block():
    """Smoke test for the TransformerBlock class"""

    def mlp_factory():
        return MLP(input_size=128, output_size=128, inner_size=512)

    def attention_factory():
        return SelfAttention(input_size=128, num_heads_q=8, num_heads_kv=2, head_dim=32)

    block = TransformerBlock(
        input_size=128, mlp_factory=mlp_factory, attention_factory=attention_factory
    )
    x = torch.randn(3, 10, 128)
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
        embedding_size=128,
    )
    vocab_size = 1024
    model = TransformerModel(vocab_size=vocab_size, seed=1337, config=config)
    x = torch.randint(0, vocab_size, (3, 10))
    y = model(x)
    assert y.shape == (3, 10, vocab_size)
    assert not torch.isnan(y).any()
