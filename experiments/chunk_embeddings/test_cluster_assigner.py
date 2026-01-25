"""
Test ClusterAssigner on a single training blob.

Verifies:
1. Output shapes are correct
2. Cluster IDs are in valid range
3. CAUSALITY: cluster_id at position p only depends on tokens 0..p-1
"""

import json
import time
import torch
import tiktoken

from cluster_assigner import ClusterAssigner


def load_blob_texts(blob_path: str, max_texts: int = 10) -> list[str]:
    """Load texts from a single JSONL blob."""
    texts = []
    with open(blob_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_texts:
                break
            data = json.loads(line)
            texts.append(data["text"])
    return texts


def test_basic_assignment():
    """Test that cluster assignment works on a single blob."""
    print("=" * 60)
    print("TEST: Basic cluster assignment on training blob")
    print("=" * 60)

    start_time = time.time()

    # Load assigner with 1M centroids
    centroid_path = "centroids_1m_w32_slimpajama.pt"
    assigner = ClusterAssigner(centroid_path, window_size=32)
    load_time = time.time() - start_time
    print(f"\nAssigner loaded in {load_time:.1f}s")

    # Load one blob
    blob_path = "../../data/slimpajama_train/blob_00000.jsonl"
    texts = load_blob_texts(blob_path, max_texts=5)
    print(f"\nLoaded {len(texts)} texts from {blob_path}")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Test each text
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text)
        print(f"\nText {i}: {len(token_ids)} tokens, first 50 chars: {text[:50]!r}...")

        # Assign with stride=32 for speed
        assign_start = time.time()
        cluster_ids = assigner.assign_single(token_ids, stride=32)
        assign_time = time.time() - assign_start

        print(f"  Assigned {len(cluster_ids)} positions in {assign_time:.3f}s")
        print(
            f"  Cluster IDs range: [{cluster_ids.min().item()}, {cluster_ids.max().item()}]"
        )
        print(f"  First 10 cluster IDs: {cluster_ids[:10].tolist()}")

        # Verify valid range
        assert cluster_ids.min() >= 0, "Cluster ID < 0"
        assert cluster_ids.max() < assigner.num_centroids, "Cluster ID >= num_centroids"

    print("\n✓ Basic assignment test PASSED")
    return time.time() - start_time


def test_causality():
    """
    Test that cluster_id at position p does NOT depend on tokens at position >= p.

    Method: For the same prefix tokens[0:p], the cluster_id at position p should be
    identical regardless of what tokens come after position p.
    """
    print("\n" + "=" * 60)
    print("TEST: Causality verification (no future leakage)")
    print("=" * 60)

    start_time = time.time()

    centroid_path = "centroids_1m_w32_slimpajama.pt"
    assigner = ClusterAssigner(centroid_path, window_size=32, compile_embedder=False)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Create a base sequence
    base_text = "The quick brown fox jumps over the lazy dog. " * 10
    base_tokens = tokenizer.encode(base_text)
    print(f"\nBase sequence: {len(base_tokens)} tokens")

    # Test position p = 50 (well past window_size)
    test_position = 50

    # Get cluster_id for position 50 with full sequence
    cluster_id_full = assigner.assign_single(base_tokens, stride=1)[test_position]

    # Now modify tokens AFTER position 50 and verify cluster_id is unchanged
    modified_tokens = base_tokens.copy()
    for j in range(test_position, len(modified_tokens)):
        modified_tokens[j] = 0  # Replace with padding token

    cluster_id_modified = assigner.assign_single(modified_tokens, stride=1)[
        test_position
    ]

    print(f"\nPosition {test_position}:")
    print(f"  Cluster ID with full sequence:     {cluster_id_full.item()}")
    print(f"  Cluster ID with modified future:   {cluster_id_modified.item()}")

    assert cluster_id_full == cluster_id_modified, (
        f"CAUSALITY VIOLATION: cluster_id changed when modifying future tokens! "
        f"full={cluster_id_full.item()}, modified={cluster_id_modified.item()}"
    )

    # Also test: truncated sequence should give same result
    truncated_tokens = base_tokens[: test_position + 1]
    cluster_id_truncated = assigner.assign_single(truncated_tokens, stride=1)[
        test_position
    ]
    print(f"  Cluster ID with truncated seq:     {cluster_id_truncated.item()}")

    assert cluster_id_full == cluster_id_truncated, (
        f"CAUSALITY VIOLATION: cluster_id differs for truncated sequence! "
        f"full={cluster_id_full.item()}, truncated={cluster_id_truncated.item()}"
    )

    # Test position 0 (should be sentinel/0 since no context)
    cluster_id_pos0 = assigner.assign_single(base_tokens, stride=1)[0]
    print(f"\nPosition 0 (no context): cluster_id = {cluster_id_pos0.item()}")

    print("\n✓ Causality test PASSED - no future leakage detected")
    return time.time() - start_time


def test_stride_coverage():
    """Test that stride parameter works correctly."""
    print("\n" + "=" * 60)
    print("TEST: Stride parameter")
    print("=" * 60)

    centroid_path = "centroids_1m_w32_slimpajama.pt"
    assigner = ClusterAssigner(centroid_path, window_size=32, compile_embedder=False)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = "Hello world! " * 50
    token_ids = tokenizer.encode(text)
    print(f"\nSequence length: {len(token_ids)} tokens")

    for stride in [1, 8, 32]:
        cluster_ids, positions = assigner.assign_single(
            token_ids, stride=stride, return_positions=True
        )
        expected_positions = list(range(0, len(token_ids), stride))
        print(f"  Stride {stride}: {len(cluster_ids)} positions assigned")
        assert len(cluster_ids) == len(
            expected_positions
        ), f"Wrong count for stride {stride}"
        assert (
            positions.tolist() == expected_positions
        ), f"Wrong positions for stride {stride}"

    print("\n✓ Stride test PASSED")


def test_model_integration():
    """Test that TransformerModel works with chunk embeddings enabled."""
    print("\n" + "=" * 60)
    print("TEST: Model integration with chunk embeddings")
    print("=" * 60)

    import sys

    sys.path.insert(0, "../..")  # Add root to path
    import transformer

    # Need CUDA for flash attention
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return
    torch.set_default_device("cuda")

    # Create small model with chunk embeddings enabled
    config = transformer.TransformerConfig(
        num_layers=2,
        num_heads=4,
        embedding_size=128,
        chunk_embeddings=True,
        chunk_num_clusters=1000,  # Small for testing
        chunk_init_scale=0.0,  # Zero-init
    )

    vocab_size = 1000
    model = transformer.TransformerModel(vocab_size, config)
    model.eval()  # Don't need training mode for this test

    print(f"\nModel created with {model.num_parameters():,} parameters")
    print(
        f"Chunk embedding table: {config.chunk_num_clusters} x {config.embedding_size}"
    )

    # Create dummy input
    batch_size, seq_len = 2, 64
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int32)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    cluster_ids = torch.randint(
        0, config.chunk_num_clusters, (batch_size, seq_len), dtype=torch.long
    )

    # Test forward pass without cluster_ids (should work)
    with torch.no_grad():
        logits_no_chunk = model(inputs, cluster_ids=None, use_optimized=False)
        print(f"\nForward (no cluster_ids): {logits_no_chunk.shape}")

    # Test forward pass with cluster_ids
    with torch.no_grad():
        logits_with_chunk = model(inputs, cluster_ids=cluster_ids, use_optimized=False)
        print(f"Forward (with cluster_ids): {logits_with_chunk.shape}")

    # With zero-init, outputs should be identical
    diff = (logits_no_chunk - logits_with_chunk).abs().max().item()
    print(f"Max difference (should be 0 for zero-init): {diff:.2e}")
    assert diff < 1e-5, f"Expected near-zero diff with zero-init, got {diff}"

    # Test compute_loss
    loss = model.compute_loss(inputs, targets, cluster_ids)
    print(f"Loss: {loss.item():.4f}")
    assert loss.isfinite(), "Loss is not finite"

    print("\n✓ Model integration test PASSED")


if __name__ == "__main__":
    total_start = time.time()

    basic_time = test_basic_assignment()
    causality_time = test_causality()
    test_stride_coverage()
    test_model_integration()

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)
