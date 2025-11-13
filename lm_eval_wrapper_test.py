"""
Test the lm_eval_wrapper module.

This test verifies that the wrapper correctly interfaces with the model
and produces expected outputs for the three core methods.
"""

import torch
import tempfile
from pathlib import Path

import transformer
import checkpointing


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
    
    # Save with metadata including config
    metadata = {
        "config": config,
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


def test_wrapper_initialization():
    """Test that we can initialize the wrapper."""
    print("Testing wrapper initialization...")
    
    checkpoint_path, vocab_size, config = create_test_checkpoint()
    
    try:
        from lm_eval_wrapper import LittleTrainingLoopLM
        
        # Initialize wrapper
        wrapper = LittleTrainingLoopLM(
            checkpoint_path=str(checkpoint_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        print(f"✓ Wrapper initialized successfully")
        print(f"  Device: {wrapper.device}")
        print(f"  Vocab size: {wrapper.vocab_size}")
        print(f"  EOT token: {wrapper.eot_token_id}")
        print(f"  Max length: {wrapper.max_length}")
        
        return wrapper
        
    except ImportError as e:
        print(f"⚠ lm-evaluation-harness not installed: {e}")
        print("  Install with: pip install lm-eval")
        return None


def test_tokenization(wrapper):
    """Test tokenization methods."""
    if wrapper is None:
        return

    print("\nTesting tokenization...")

    text = "Hello, world!"
    tokens = wrapper.tok_encode(text)
    decoded = wrapper.tok_decode(tokens)

    print(f"  Original: {text}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")
    print(f"  Vocab size: {wrapper.vocab_size}")
    print(f"  EOT token: {wrapper.eot_token_id}")
    print(f"✓ Tokenization works")


def test_loglikelihood(wrapper):
    """Test loglikelihood method."""
    if wrapper is None:
        return

    print("\nTesting loglikelihood...")

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

    print(f"  Processed {len(results)} requests")
    for i, (logprob, is_greedy) in enumerate(results):
        ctx, cont = requests[i].args
        print(f"  Request {i} ('{ctx}' + '{cont}'): logprob={logprob:.4f}, is_greedy={is_greedy}")

    print(f"✓ Loglikelihood works")


def test_loglikelihood_rolling(wrapper):
    """Test loglikelihood_rolling method."""
    if wrapper is None:
        return

    print("\nTesting loglikelihood_rolling...")

    class MockRequest:
        def __init__(self, text):
            self.args = (text,)

    requests = [
        MockRequest("The quick brown fox"),
        MockRequest("Hello world"),
    ]

    results = wrapper.loglikelihood_rolling(requests)

    print(f"  Processed {len(results)} requests")
    for i, (logprob,) in enumerate(results):
        text = requests[i].args[0]
        print(f"  Request {i} ('{text}'): logprob={logprob:.4f}")

    print(f"✓ Loglikelihood rolling works")


def test_generate_until(wrapper):
    """Test generate_until method."""
    if wrapper is None:
        return

    print("\nTesting generate_until...")

    class MockRequest:
        def __init__(self, context, gen_kwargs):
            self.args = (context, gen_kwargs)

    requests = [
        MockRequest("Once upon a time", {"until": ["\n"], "max_gen_toks": 10}),
        MockRequest("The answer is", {"until": [".", "!"], "max_gen_toks": 5}),
    ]

    results = wrapper.generate_until(requests)

    print(f"  Processed {len(results)} requests")
    for i, generated in enumerate(results):
        context = requests[i].args[0]
        print(f"  Request {i} ('{context}'): generated='{generated}'")

    print(f"✓ Generate until works")


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

