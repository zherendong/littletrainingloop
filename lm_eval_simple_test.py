"""
Simple test to verify lm-eval integration works.

This test verifies that our wrapper is compatible with lm-eval's API
without actually running full benchmarks (which require dataset downloads).
"""

import torch
import tempfile
from pathlib import Path

import dataclasses

import transformer
import checkpointing
import language_model_basics
from lm_eval_wrapper import LittleTrainingLoopLM


def create_test_checkpoint():
    """Create a test checkpoint with full tiktoken vocab."""
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab

    config = transformer.TransformerConfig(
        num_layers=2,
        num_heads=2,
        embedding_size=64,
        mlp_inner_size=256,
        zheren_init=False,
    )

    model = transformer.TransformerModel(vocab_size, config)

    tmpdir = tempfile.mkdtemp()
    checkpoint_path = Path(tmpdir) / "test_model.pt"

    training_config = language_model_basics.LanguageModelTrainingConfig(
        vocab_size=vocab_size,
        model_config=config,
    )

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

    return checkpoint_path


def test_lm_eval_api_compatibility():
    """Test that our wrapper implements the lm-eval API correctly."""
    checkpoint_path = create_test_checkpoint()

    # Initialize wrapper
    wrapper = LittleTrainingLoopLM(
        checkpoint_path=str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
    )

    # Check that wrapper has all required methods
    required_methods = [
        "loglikelihood",
        "loglikelihood_rolling",
        "generate_until",
        "tok_encode",
        "tok_decode",
    ]
    for method in required_methods:
        assert hasattr(wrapper, method), f"Missing required method: {method}"

    # Check required properties
    required_props = [
        "eot_token_id",
        "max_length",
        "max_gen_toks",
        "batch_size",
        "device",
    ]
    for prop in required_props:
        assert hasattr(wrapper, prop), f"Missing required property: {prop}"

    # Basic sanity checks
    assert wrapper.vocab_size > 0
    assert wrapper.max_length > 0
    assert isinstance(wrapper.eot_token_id, int)


def test_lm_eval_request_format():
    """Test that our wrapper handles lm-eval request format correctly."""
    print("\nTesting lm-eval request format handling...")

    checkpoint_path = create_test_checkpoint()
    wrapper = LittleTrainingLoopLM(
        checkpoint_path=str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
    )

    # Create mock requests in lm-eval format
    class MockRequest:
        def __init__(self, *args):
            self.args = args

    # Test loglikelihood
    print("  Testing loglikelihood...")
    ll_requests = [
        MockRequest("The cat", " sat"),
        MockRequest("Hello", " world"),
    ]
    ll_results = wrapper.loglikelihood(ll_requests)
    assert len(ll_results) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ll_results)
    print(f"    ✓ Returns correct format: {ll_results}")

    # Test loglikelihood_rolling
    print("  Testing loglikelihood_rolling...")
    llr_requests = [
        MockRequest("The quick brown fox"),
    ]
    llr_results = wrapper.loglikelihood_rolling(llr_requests)
    assert len(llr_results) == 1
    assert all(isinstance(r, tuple) and len(r) == 1 for r in llr_results)
    print(f"    ✓ Returns correct format: {llr_results}")

    # Test generate_until
    print("  Testing generate_until...")
    gen_requests = [
        MockRequest("Once upon a time", {"until": ["\n"], "max_gen_toks": 5}),
    ]
    gen_results = wrapper.generate_until(gen_requests)
    assert len(gen_results) == 1
    assert all(isinstance(r, str) for r in gen_results)
    print(f"    ✓ Returns correct format: ['{gen_results[0]}']")

    print("\n  ✓ All request formats handled correctly!")


if __name__ == "__main__":
    print("=" * 60)
    print("LM-Eval Simple Integration Test")
    print("=" * 60)

    all_ok = True
    try:
        test_lm_eval_api_compatibility()
        test_lm_eval_request_format()
    except AssertionError:
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All tests passed!")
        print("\nThe wrapper is compatible with lm-evaluation-harness API.")
        print("You can now use it with lm-eval's simple_evaluate() function")
        print("or any other lm-eval evaluation methods.")
    else:
        print("✗ Some tests failed.")
    print("=" * 60)

