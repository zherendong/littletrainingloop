"""
Validate embedding similarity assumptions.

Tests:
1. Similar strings (shifted by 1 token) should have similar embeddings
2. Very different strings should have distant embeddings

Usage:
    python validate_embeddings.py --seed 42
    python validate_embeddings.py --seed 123
"""

import argparse
import json
import glob
import random
from sentence_transformers import SentenceTransformer
import numpy as np


def load_random_slimpajama_texts(n: int = 3, seed: int = 42) -> list[str]:
    """Load n random text samples from SlimPajama."""
    random.seed(seed)

    # Find available JSONL files
    files = glob.glob("../../data/slimpajama_train/*.jsonl")
    if not files:
        raise FileNotFoundError("No SlimPajama files found in data/slimpajama_train/")

    # Pick a random file
    file = random.choice(files)
    print(f"Loading from: {file}")

    # Load all texts from that file
    texts = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])

    # Pick n random texts (at least 200 chars for meaningful context)
    long_texts = [t for t in texts if len(t) > 200]
    return random.sample(long_texts, min(n, len(long_texts)))


def tokenize_text(text: str, tokenizer) -> list[int]:
    """Tokenize text using the embedding model's tokenizer."""
    return tokenizer.encode(text)


def create_windows(tokens: list[int], tokenizer, window_size: int = 32) -> list[str]:
    """Create overlapping windows from tokens, return as text."""
    windows = []
    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i : i + window_size]
        # Decode back to text for embedding
        window_text = tokenizer.decode(window_tokens)
        windows.append(window_text)
    return windows


def main(seed: int = 42):
    print("Loading embeddinggemma-300m...")
    model = SentenceTransformer("google/embeddinggemma-300m")
    tokenizer = model.tokenizer
    print(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Load random texts
    print(f"\n=== Loading SlimPajama samples (seed={seed}) ===")
    texts = load_random_slimpajama_texts(n=3, seed=seed)
    for i, text in enumerate(texts):
        print(f"\nText {i+1} ({len(text)} chars): {text[:100]}...")

    # Test 1: Similar strings (shifted by 1 token)
    print("\n" + "=" * 60)
    print("TEST 1: Similarity of shifted windows (should be HIGH)")
    print("=" * 60)

    text = texts[0]
    tokens = tokenize_text(text, tokenizer)
    print(f"Text has {len(tokens)} tokens")

    # Create consecutive windows
    windows = create_windows(tokens, tokenizer, window_size=32)[:10]  # First 10 windows
    print(f"Created {len(windows)} windows")

    # Embed all windows
    embeddings = model.encode(windows)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute similarities between consecutive windows
    print("\nCosine similarity between consecutive windows:")
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        print(f"  Window {i} vs {i+1}: {sim:.4f}")

    # Test 2: Different strings (should be distant)
    print("\n" + "=" * 60)
    print("TEST 2: Similarity of different texts (should be LOW)")
    print("=" * 60)

    # Get first 32-token window from each text
    different_windows = []
    for i, text in enumerate(texts):
        tokens = tokenize_text(text, tokenizer)
        if len(tokens) >= 32:
            window_text = tokenizer.decode(tokens[:32])
            different_windows.append(window_text)
            print(f"Window {i}: {window_text[:60]}...")

    # Embed and compare
    diff_embeddings = model.encode(different_windows)

    print("\nCosine similarity between different texts:")
    for i in range(len(diff_embeddings)):
        for j in range(i + 1, len(diff_embeddings)):
            sim = np.dot(diff_embeddings[i], diff_embeddings[j]) / (
                np.linalg.norm(diff_embeddings[i]) * np.linalg.norm(diff_embeddings[j])
            )
            print(f"  Text {i} vs Text {j}: {sim:.4f}")

    # Test 3: Compare consecutive vs non-consecutive
    print("\n" + "=" * 60)
    print("TEST 3: Consecutive vs non-consecutive windows")
    print("=" * 60)

    # Use embeddings from Test 1
    consecutive_sims = []
    non_consecutive_sims = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if j == i + 1:
                consecutive_sims.append(sim)
            else:
                non_consecutive_sims.append(sim)

    print(
        f"Consecutive (shift=1):     mean={np.mean(consecutive_sims):.4f}, std={np.std(consecutive_sims):.4f}"
    )
    print(
        f"Non-consecutive (shift>1): mean={np.mean(non_consecutive_sims):.4f}, std={np.std(non_consecutive_sims):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
