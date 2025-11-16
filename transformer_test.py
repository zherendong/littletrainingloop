"""
Test script for the transformer.py module.

Mostly smoke tests
"""

import torch
import torch.nn.functional as F
from torch.utils import flop_counter
import pytest
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
        67161120,
        67461120,
        100584480,
    ]  # depending on use of flash attention / architecture
    assert sum(p.numel() for p in model.parameters()) in [
        1377280,
        1377536,
        1934592,
    ]  # depending on use of flash attention / architecture

    # now count with backward
    flops = flop_counter.FlopCounterMode(depth=4)
    with flops:
        y = model(x)
        loss = y.sum()
        loss.backward()
    assert flops.get_total_flops() in [201483360, 220385280, 368914560]


def test_mini_transformer_smoketest():
    """Smoke test for the mini transformer"""
    config = transformer.transformer_config_registry.get("mini-transformer-1")
    vocab_size = 1024
    model = TransformerModel(vocab_size=vocab_size, config=config)
    x = torch.randint(0, vocab_size, (3, 10), dtype=torch.int32)
    y = model(x)
    assert y.shape == (3, 10, vocab_size)
    assert not torch.isnan(y).any()


def _make_small_transformer(vocab_size: int = 128) -> TransformerModel:
    config = TransformerConfig(
        num_layers=1,
        num_heads=4,
        num_heads_kv=2,
        head_dim=32,
        mlp_inner_size=128,
        embedding_size=64,
    )
    return TransformerModel(vocab_size=vocab_size, config=config)


def test_get_logits_matches_forward():
    """get_logits should match forward() output."""
    vocab_size = 128
    model = _make_small_transformer(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (1, 5), dtype=torch.long)

    logits_forward = model(x)
    logits_get = model.get_logits(x)

    assert logits_get.shape == logits_forward.shape
    assert torch.allclose(logits_get, logits_forward, atol=1e-5, rtol=1e-5)


def test_compute_token_logprobs_matches_manual():
    """compute_token_logprobs should match a manual logprob computation."""
    vocab_size = 128
    model = _make_small_transformer(vocab_size=vocab_size)
    context_len = 4
    target_len = 3

    input_ids = torch.randint(0, vocab_size, (1, context_len), dtype=torch.long)
    target_ids = torch.randint(0, vocab_size, (1, target_len), dtype=torch.long)

    # Manual computation based on model logits
    full_input = torch.cat([input_ids, target_ids], dim=1)
    logits = model.get_logits(full_input)

    context_len = input_ids.shape[1]
    target_len = target_ids.shape[1]
    prediction_logits = logits[:, context_len - 1 : context_len + target_len - 1, :]
    log_probs = F.log_softmax(prediction_logits, dim=-1)
    manual_logprobs = log_probs[0, torch.arange(target_len), target_ids[0]]

    token_logprobs = model.compute_token_logprobs(input_ids, target_ids)

    assert token_logprobs.shape == (1, target_len)
    assert torch.allclose(
        token_logprobs[0], manual_logprobs, atol=1e-4, rtol=1e-4
    )


def test_compute_token_logprobs_empty_context_raises():
    """compute_token_logprobs should error on empty context."""
    vocab_size = 128
    model = _make_small_transformer(vocab_size=vocab_size)

    input_ids = torch.empty((1, 0), dtype=torch.long)
    target_ids = torch.randint(0, vocab_size, (1, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="at least 1 context token"):
        model.compute_token_logprobs(input_ids, target_ids)


def test_is_greedy_generation_true_and_false():
    """is_greedy_generation should detect greedy vs non-greedy continuations."""
    vocab_size = 128
    model = _make_small_transformer(vocab_size=vocab_size)
    context_len = 4
    continuation_len = 3

    input_ids = torch.randint(0, vocab_size, (1, context_len), dtype=torch.long)

    # Build a greedy continuation by autoregressive greedy decoding
    continuation_tokens = []
    cur_seq = input_ids.clone()
    for _ in range(continuation_len):
        logits = model.get_logits(cur_seq)
        next_token = logits[:, -1, :].argmax(dim=-1)  # [1]
        continuation_tokens.append(next_token)
        cur_seq = torch.cat([cur_seq, next_token.unsqueeze(1)], dim=1)

    continuation_ids = torch.stack(continuation_tokens, dim=1)  # [1, continuation_len]

    is_greedy = model.is_greedy_generation(input_ids, continuation_ids)
    assert is_greedy.shape == (1,)
    assert is_greedy.item() is True

    # Modify one token to make it non-greedy
    non_greedy = continuation_ids.clone()
    non_greedy[0, 0] = (non_greedy[0, 0] + 1) % vocab_size

    is_greedy_ng = model.is_greedy_generation(input_ids, non_greedy)
    assert is_greedy_ng.shape == (1,)
    assert is_greedy_ng.item() is False


def test_is_greedy_generation_empty_context_raises():
    """is_greedy_generation should error on empty context."""
    vocab_size = 128
    model = _make_small_transformer(vocab_size=vocab_size)

    input_ids = torch.empty((1, 0), dtype=torch.long)
    continuation_ids = torch.randint(0, vocab_size, (1, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="at least 1 context token"):
        model.is_greedy_generation(input_ids, continuation_ids)
