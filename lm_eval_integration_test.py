"""
Integration test with lm-evaluation-harness library.

This test verifies that our wrapper works correctly with the actual
lm-evaluation-harness framework by running a simple evaluation task.
"""

import torch
import tempfile
from pathlib import Path

import transformer
import checkpointing
from lm_eval_wrapper import LittleTrainingLoopLM

# Import lm-eval library components
from lm_eval import simple_evaluate
try:
    from lm_eval.models import get_model
except ImportError:
    # get_model might not be available in all versions
    get_model = None


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
    
    return checkpoint_path


def test_model_registration():
    """Test that our model is properly registered with lm-eval."""
    if get_model is None:
        # Skip if this lm-eval version does not expose get_model
        import pytest

        pytest.skip("get_model not available in this lm-eval version")

    _ = create_test_checkpoint()

    # Try to get our model through lm-eval's registry
    # Our model should be registered as "littletrainingloop"
    model_cls = get_model("littletrainingloop")
    assert model_cls is not None


def test_simple_evaluation():
    """Test running a simple evaluation task."""
    print("\nTesting simple evaluation with lm-eval...")

    checkpoint_path = create_test_checkpoint()

    # Initialize our wrapper directly
    wrapper = LittleTrainingLoopLM(
        checkpoint_path=str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
    )

    # Run a very simple evaluation using lm-eval's simple_evaluate
    # We'll use a tiny subset of a task to keep it fast
    results = simple_evaluate(
        model=wrapper,
        tasks=["hellaswag"],  # Simple multiple-choice task
        num_fewshot=0,  # Zero-shot
        limit=5,  # Only evaluate on 5 examples for speed
        device=wrapper.device,
    )

    # Check that we got some results in the expected structure
    assert "results" in results
    assert "hellaswag" in results["results"]
    hellaswag_results = results["results"]["hellaswag"]
    assert isinstance(hellaswag_results, dict)


def test_multiple_tasks():
    """Test running multiple tasks."""
    print("\nTesting multiple tasks...")

    checkpoint_path = create_test_checkpoint()

    wrapper = LittleTrainingLoopLM(
        checkpoint_path=str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1,
    )

    # Test with multiple simple tasks
    results = simple_evaluate(
        model=wrapper,
        tasks=["hellaswag", "arc_easy"],
        num_fewshot=0,
        limit=3,  # Very small for speed
        device=wrapper.device,
    )

    assert "results" in results
    tasks_run = list(results["results"].keys())
    assert "hellaswag" in tasks_run
    assert "arc_easy" in tasks_run


if __name__ == "__main__":
    print("=" * 60)
    print("LM-Evaluation-Harness Integration Test")
    print("=" * 60)
    
    # Run tests
    test1 = test_model_registration()
    test2 = test_simple_evaluation()
    test3 = test_multiple_tasks()
    
    print("\n" + "=" * 60)
    if test1 and test2 and test3:
        print("All integration tests passed! ✓")
    else:
        print("Some tests failed. See output above.")
    print("=" * 60)

