"""
Test the lm_eval_wrapper module.

This test verifies that the wrapper correctly interfaces with the model
and produces expected outputs for the three core methods.
"""

import torch
import tempfile
from pathlib import Path

import dataclasses
import pytest

import transformer
import checkpointing
import language_model_basics


def create_test_checkpoint():
    """Create a small test checkpoint for testing."""
    # Use tiktoken vocab size to match the tokenizer
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab  # 100277

    config = transformer.TransformerConfig(
        num_layers=2,
        num_heads=2,
        embedding_size=64,
        mlp_inner_size=256,
        zheren_init=False,  # Disable custom init for faster testing
    )

    model = transformer.TransformerModel(vocab_size, config)

    # Create a temporary checkpoint
    tmpdir = tempfile.mkdtemp()
    checkpoint_path = Path(tmpdir) / "test_model.pt"

    training_config = language_model_basics.LanguageModelTrainingConfig(
        vocab_size=vocab_size,
        model_config=config,
    )

    # Save with metadata mirroring a training checkpoint
    metadata = {
        "config": dataclasses.asdict(training_config),
        "vocab_size": vocab_size,
        "step": 0,
        "epoch": 0,
    }

    checkpointing.save_checkpoint(
        model=model,
        path=checkpoint_path,
        metadata=metadata,
    )

    return checkpoint_path, vocab_size, config


@pytest.fixture
def wrapper():
    """Fixture that provides an initialized LittleTrainingLoopLM wrapper."""
    checkpoint_path, vocab_size, config = create_test_checkpoint()

    try:
        from lm_eval_wrapper import LittleTrainingLoopLM
    except ImportError as e:
        pytest.skip(f"lm-evaluation-harness not installed: {e}")

    return LittleTrainingLoopLM(
        checkpoint_path=str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def test_wrapper_initialization(wrapper):
    """Test that we can initialize the wrapper."""
    # Basic sanity checks
    assert wrapper.vocab_size > 0
    assert wrapper.max_length > 0
    assert wrapper.eot_token_id >= 0


def test_tokenization(wrapper):
    """Test tokenization round-trip and vocab alignment."""
    text = "Hello, world!"
    tokens = wrapper.tok_encode(text)
    decoded = wrapper.tok_decode(tokens)

    # Round-trip should not be empty and should contain the original text as a substring
    assert isinstance(tokens, list) and len(tokens) > 0
    assert isinstance(decoded, str) and decoded != ""
    assert "Hello" in decoded

    # Token IDs should be within vocab range
    assert all(0 <= t < wrapper.vocab_size for t in tokens)


def test_loglikelihood(wrapper):
    """Test loglikelihood method returns well-formed outputs."""

    # Create a mock request object
    class MockRequest:
        def __init__(self, context, continuation):
            self.args = (context, continuation)

    requests = [
        MockRequest("The cat sat on the", " mat"),
        MockRequest("Hello", " world"),
        MockRequest("", "Test"),  # Empty context
    ]

    results = wrapper.loglikelihood(requests)

    assert len(results) == len(requests)
    for (logprob, is_greedy) in results:
        assert isinstance(logprob, float)
        assert isinstance(is_greedy, bool)


def test_loglikelihood_rolling(wrapper):
    """Test loglikelihood_rolling method returns well-formed outputs."""

    class MockRequest:
        def __init__(self, text):
            self.args = (text,)

    requests = [
        MockRequest("The quick brown fox"),
        MockRequest("Hello world"),
    ]

    results = wrapper.loglikelihood_rolling(requests)

    assert len(results) == len(requests)
    for (logprob,) in results:
        assert isinstance(logprob, float)


def test_generate_until(wrapper):
    """Test generate_until method returns strings for each request."""

    class MockRequest:
        def __init__(self, context, gen_kwargs):
            self.args = (context, gen_kwargs)

    requests = [
        MockRequest("Once upon a time", {"until": ["\n"], "max_gen_toks": 10}),
        MockRequest("The answer is", {"until": [".", "!"], "max_gen_toks": 5}),
    ]

    results = wrapper.generate_until(requests)

    assert len(results) == len(requests)
    for generated in results:
        assert isinstance(generated, str)


if __name__ == "__main__":
    wrapper = test_wrapper_initialization()
    test_tokenization(wrapper)
    test_loglikelihood(wrapper)
    test_loglikelihood_rolling(wrapper)
    test_generate_until(wrapper)
    
    print("\n" + "=" * 50)
    if wrapper is not None:
        print("All wrapper tests passed! ✓")
    else:
        print("Wrapper tests skipped (lm-eval not installed)")
    print("=" * 50)
