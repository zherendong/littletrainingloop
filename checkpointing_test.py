"""
Test checkpoint saving and loading.
"""

import torch
import tempfile
from pathlib import Path

import checkpointing
import transformer
import language_model_basics


def test_save_and_load_checkpoint():
    """Test that we can save and load a checkpoint."""
    print("Testing checkpoint save/load...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a small model
    vocab_size = 1000
    config = transformer.TransformerConfig(
        num_layers=2,
        num_heads=4,
        embedding_size=128,
        mlp_inner_size=512,
    )

    # Create model
    model = transformer.TransformerModel(vocab_size, config)
    model.to(device)
    
    # Get initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        
        metadata = {
            "step": 100,
            "epoch": 1,
            "test_value": "hello",
        }
        
        checkpointing.save_checkpoint(
            model=model,
            path=checkpoint_path,
            metadata=metadata,
        )
        
        # Load checkpoint
        loaded = checkpointing.load_checkpoint(
            path=checkpoint_path,
            vocab_size=vocab_size,
            model_config=config,
            device=device,
        )
        
        loaded_model = loaded["model"]
        loaded_metadata = loaded["metadata"]
        
        # Verify metadata
        assert loaded_metadata["step"] == 100
        assert loaded_metadata["epoch"] == 1
        assert loaded_metadata["test_value"] == "hello"
        print("✓ Metadata loaded correctly")
        
        # Verify weights match
        for name, param in loaded_model.named_parameters():
            assert torch.allclose(param, initial_weights[name], rtol=1e-5, atol=1e-8), \
                f"Parameter {name} doesn't match"
        print("✓ Model weights loaded correctly")
    
    print("✓ Checkpoint save/load test passed!")


def test_inference_methods():
    """Test the new inference methods."""
    print("\nTesting inference methods...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a small model
    vocab_size = 100
    config = transformer.TransformerConfig(
        num_layers=2,
        num_heads=2,
        embedding_size=64,
        mlp_inner_size=256,
    )

    model = transformer.TransformerModel(vocab_size, config)
    model.to(device)
    model.eval()

    # Test get_logits
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    logits = model.get_logits(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("✓ get_logits works correctly")
    
    # Test compute_token_logprobs
    context_len = 5
    target_len = 3
    context_ids = input_ids[:, :context_len]
    target_ids = input_ids[:, context_len:context_len + target_len]
    
    logprobs = model.compute_token_logprobs(context_ids, target_ids)
    assert logprobs.shape == (batch_size, target_len)
    assert torch.all(logprobs <= 0), "Log probabilities should be <= 0"
    print("✓ compute_token_logprobs works correctly")
    
    # Test is_greedy_generation
    # First, get the actual greedy continuation
    with torch.no_grad():
        full_logits = model.forward(context_ids)
        greedy_next_tokens = full_logits[:, -1:, :].argmax(dim=-1)
    
    is_greedy = model.is_greedy_generation(context_ids, greedy_next_tokens)
    assert is_greedy.all(), "Greedy tokens should be detected as greedy"
    print("✓ is_greedy_generation works correctly")
    
    # Test with non-greedy tokens
    non_greedy_tokens = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    is_greedy = model.is_greedy_generation(context_ids, non_greedy_tokens)
    # Most random tokens should not be greedy (though there's a small chance)
    print(f"  Non-greedy detection: {(~is_greedy).sum()}/{batch_size} detected as non-greedy")
    
    print("✓ All inference methods test passed!")


if __name__ == "__main__":
    test_save_and_load_checkpoint()
    test_inference_methods()
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

