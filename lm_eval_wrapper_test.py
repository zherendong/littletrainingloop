"""
Test the lm_eval_wrapper module.

This test verifies that the wrapper correctly interfaces with the model
and produces expected outputs for the three core methods.
"""

import math
import os

import pytest

import transformer
import language_model_basics
import tiktoken
import lm_eval_wrapper
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for lm_eval_wrapper tests", allow_module_level=True)

torch.set_default_device("cuda")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def create_test_model() -> tuple[
    language_model_basics.LanguageModel,
    language_model_basics.LanguageModelTrainingConfig,
]:
    """Create a small test model for testing."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab  # 100277

    config = transformer.TransformerConfig(
        num_layers=2,
        num_heads=2,
        embedding_size=64,
        mlp_inner_size=256,
    )

    model = transformer.TransformerModel(vocab_size, config)

    training_config = language_model_basics.LanguageModelTrainingConfig(
        vocab_size=vocab_size,
        model_config=config,
    )

    return model, training_config


@pytest.fixture(scope="module")
def wrapper():
    """Fixture that provides an initialized LittleTrainingLoopLM wrapper."""
    model, config = create_test_model()
    return lm_eval_wrapper.LittleTrainingLoopWrapper(
        model=model,
        config=config,
        batch_size=4,
        generate_until_max_length=10,
    )


def test_wrapper_initialization(wrapper):
    """Test that we can initialize the wrapper."""
    # Basic sanity checks
    assert wrapper.vocab_size > 0
    assert wrapper.generate_until_max_length > 0
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

    class MockRequest:
        def __init__(self, context, continuation):
            self.args = (context, continuation)

    requests = [
        MockRequest("The cat sat on the", " mat"),
        MockRequest("Hello", " world"),
    ]

    results = wrapper.loglikelihood(requests)

    assert len(results) == len(requests)
    for logprob, is_greedy in results:
        assert isinstance(logprob, float)
        assert math.exp(logprob) >= 0
        assert math.exp(logprob) <= 1
        assert isinstance(is_greedy, bool)


def test_empty_context_raises(wrapper):
    """Test that an empty context raises an error."""

    class MockRequest:
        def __init__(self, context, continuation):
            self.args = (context, continuation)

    requests = [MockRequest("", "Test")]  # Empty context

    with pytest.raises(AssertionError):
        wrapper.loglikelihood(requests)


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
    for logprob in results:
        assert isinstance(logprob, float)
        assert math.exp(logprob) >= 0
        assert math.exp(logprob) <= 1


def test_generate_until(wrapper):
    """Test generate_until method returns strings for each request."""

    class MockRequest:
        def __init__(self, context, gen_kwargs):
            self.args = (context, gen_kwargs)

    requests = [
        MockRequest("Once upon a time", {"until": ["\n"]}),
        MockRequest("The answer is", {"until": [".", "!"]}),
    ]

    results = wrapper.generate_until(requests)

    assert len(results) == len(requests)
    for generated in results:
        assert isinstance(generated, str)


def test_evaluate():
    """Test the evaluate_model function."""
    model, config = create_test_model()
    results = lm_eval_wrapper.evaluate_model(
        model=model,
        config=config,
        tasks=["humaneval", "hellaswag", "arc_easy"],
        limit=100,
        generate_until_max_length=3,
    )
    assert isinstance(results, dict)
    # print(results["results"]["humaneval"])
    # print(results["results"]["hellaswag"])
    # print(results["results"]["arc_easy"])
    # assert False
