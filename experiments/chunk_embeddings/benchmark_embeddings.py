"""Benchmark embedding models for the chunk embeddings experiment."""

import time
import argparse
from sentence_transformers import SentenceTransformer


def benchmark_model(model_name: str, batch_sizes: list[int] = None):
    """Benchmark an embedding model."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]

    print(f"=== Testing {model_name} ===")
    print("Loading model...")
    start = time.time()
    model = SentenceTransformer(model_name)
    print(f"Model loaded in {time.time() - start:.2f}s")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Test basic functionality
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]

    print("Testing encode...")
    if hasattr(model, "encode_query"):
        query_emb = model.encode_query(query)
        doc_emb = model.encode_document(documents)
    else:
        query_emb = model.encode(query)
        doc_emb = model.encode(documents)

    print(f"Query shape: {query_emb.shape}")
    print(f"Document shape: {doc_emb.shape}")

    similarities = model.similarity(query_emb, doc_emb)
    print(f"Similarities: {similarities}")

    # Speed test
    print()
    print("=== Speed Benchmark ===")

    # Generate text that is exactly 32 tokens
    # "The quick brown fox jumps over the lazy dog" repeated and truncated
    base_text = "The quick brown fox jumps over the lazy dog. "
    long_text = base_text * 10  # Make it long enough

    # Tokenize and find 32-token text (excluding bos/eos, so target 30 content tokens)
    tokens = model.tokenizer.encode(long_text)
    # tokens includes <bos> and <eos>, so we want 30 content tokens + 2 special = 32 total
    truncated_tokens = tokens[:32]  # Take first 32 tokens
    test_text = model.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    actual_tokens = len(model.tokenizer.encode(test_text))
    print(f"Test text tokens: {actual_tokens}")

    max_batch = max(batch_sizes)
    test_texts = [test_text for _ in range(max_batch)]

    # Warmup
    _ = model.encode(test_texts[:10])

    num_iterations = 3
    best_rate = 0

    for batch_size in batch_sizes:
        texts = test_texts[:batch_size]

        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = model.encode(texts)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        texts_per_sec = batch_size / avg_time
        best_rate = max(best_rate, texts_per_sec)
        print(
            f"Batch size {batch_size:3d}: {avg_time*1000:7.1f}ms, {texts_per_sec:8.1f} texts/sec"
        )

    print()
    print(f"=== 1B Embedding Estimate ({model_name}) ===")
    time_for_1b = 1_000_000_000 / best_rate
    print(f"At {best_rate:.1f} texts/sec:")
    print(f"  1B embeddings = {time_for_1b:,.0f} seconds")
    print(f"                = {time_for_1b/3600:,.1f} hours")
    print(f"                = {time_for_1b/3600/24:,.1f} days")
    print()
    print(f"With 8x GPUs: ~{time_for_1b/3600/24/8:.1f} days")

    return best_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    parser.add_argument(
        "--batch_sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256]
    )
    args = parser.parse_args()

    benchmark_model(args.model, args.batch_sizes)
