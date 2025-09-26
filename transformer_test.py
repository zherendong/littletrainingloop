"""
Test script for the transformer.py module.

Mostly smoke tests
"""

import torch
from torch.utils import flop_counter
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
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    y = mlp(x)
    assert y.shape == (3, 256)
    assert not torch.isnan(y).any()


def test_attention_fn():
    """Smoke test for the attention_fn function"""
    q = torch.randn(1, 3, 2, 4, 32, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16)
    y = attention_fn(q, k, v)
    assert y.shape == (1, 3, 2, 4, 32)
    assert not torch.isnan(y).any()


def test_self_attention():
    """Smoke test for the SelfAttention class"""
    attention = SelfAttention(
        input_size=128, num_heads_q=8, num_heads_kv=2, head_dim=32
    )
    x = torch.randn(3, 10, 128, dtype=torch.bfloat16)
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
    x = torch.randint(0, vocab_size, (3, 10))
    fcounter = flop_counter.FlopCounterMode(depth=4)
    with fcounter:
        y = model(x)
    assert y.shape == (3, 10, vocab_size)
    assert not torch.isnan(y).any()
    assert fcounter.get_total_flops() == 67461120
    assert sum(p.numel() for p in model.parameters()) == 1379328


def test_flash_nonflash_equivalence():
    """Test that flash and non-flash attention give the same results"""
    if not torch.cuda.is_available():
        return
    q = torch.normal(mean=0, std=1, size=(1, 3, 2, 4, 32), dtype=torch.bfloat16, device="cuda")
    k = torch.normal(mean=0, std=1, size=(1, 3, 2, 32), dtype=torch.bfloat16, device="cuda")
    v = torch.normal(mean=0, std=1, size=(1, 3, 2, 32), dtype=torch.bfloat16, device="cuda")
    # q_copy = q.clone()
    # q_copy = q.reshape(1, 3, 2 * 4, 32)
    # k_copy = k.clone()
    # v_copy = v.clone()
    y_flash = attention_fn(q, k, v, use_flash=True)
    y_nonflash = attention_fn(q, k, v, use_flash=False)
    torch.testing.assert_close(y_flash, y_nonflash, rtol=1e-2, atol=1e-2)
