"""
Test whether random centroids can be used instead of learned clustering.

The key insight is that in high-dimensional space (768-dim), random unit vectors
are nearly orthogonal. Similar embeddings have similar dot products with all
centroids, so they tend to have the same argmax (nearest centroid).

Usage:
    python test_random_centroids.py --num-windows 500 --seed 42
"""

import argparse
import json
import glob
import random
import numpy as np
from sentence_transformers import SentenceTransformer


def load_diverse_windows(
    data_dir: str,
    num_windows: int,
    window_size: int,
    tokenizer,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Load windows from random positions in random files.
    
    Each window comes from a different random position in a random file,
    ensuring maximum diversity in the sample.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    files = glob.glob(f"{data_dir}/*.jsonl")
    if not files:
        raise ValueError(f"No JSONL files found in {data_dir}")
    
    print(f"Found {len(files)} files in {data_dir}")
    
    windows = []
    attempts = 0
    max_attempts = num_windows * 10
    
    while len(windows) < num_windows and attempts < max_attempts:
        attempts += 1
        
        # Pick a random file
        file = random.choice(files)
        
        # Load texts from file
        with open(file, "r") as f:
            texts = [json.loads(line)["text"] for line in f]
        
        if not texts:
            continue
        
        # Pick a random text
        text = random.choice(texts)
        tokens = tokenizer.encode(text)
        
        if len(tokens) < window_size:
            continue
        
        # Pick a random position
        pos = random.randint(0, len(tokens) - window_size)
        window_tokens = tokens[pos : pos + window_size]
        window_text = tokenizer.decode(window_tokens)
        windows.append(window_text)
    
    print(f"Loaded {len(windows)} windows from {attempts} attempts")
    return windows


def run_experiment(
    model: SentenceTransformer,
    windows: list[str],
    centroid_counts: list[int],
    distances: list[int],
    seed: int = 42,
) -> dict:
    """
    Run the random centroids experiment.
    
    For each centroid count, compute what % of window pairs at each distance
    map to the same centroid.
    """
    print(f"\nEmbedding {len(windows)} windows...")
    embeddings = model.encode(windows, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    results = {}
    
    for num_centroids in centroid_counts:
        print(f"\nTesting {num_centroids:,} centroids...")
        
        # Generate random centroids (deterministic given seed)
        np.random.seed(seed)
        centroids = np.random.randn(num_centroids, embeddings.shape[1]).astype(np.float32)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        # Find nearest centroid for each embedding
        # Process in batches to avoid memory issues
        batch_size = 1000
        cluster_ids = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            batch_ids = np.argmax(batch @ centroids.T, axis=1)
            cluster_ids.extend(batch_ids)
        cluster_ids = np.array(cluster_ids)
        
        # For diverse windows, "distance" means index distance in our random sample
        # We compare pairs of windows to see if they share a centroid
        # Since windows are independently sampled, we expect low overlap
        
        # Compute pairwise same-cluster rate for random pairs
        num_pairs = min(10000, len(windows) * (len(windows) - 1) // 2)
        same_cluster_count = 0
        
        np.random.seed(seed + num_centroids)  # Different seed for sampling pairs
        for _ in range(num_pairs):
            i, j = np.random.choice(len(windows), 2, replace=False)
            if cluster_ids[i] == cluster_ids[j]:
                same_cluster_count += 1
        
        same_cluster_rate = same_cluster_count / num_pairs
        
        results[num_centroids] = {
            "same_cluster_rate": same_cluster_rate,
            "unique_clusters": len(np.unique(cluster_ids)),
            "total_windows": len(windows),
        }
        
        print(f"  Same cluster rate (random pairs): {same_cluster_rate:.2%}")
        print(f"  Unique clusters used: {results[num_centroids]['unique_clusters']:,} / {num_centroids:,}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test random centroids for clustering")
    parser.add_argument("--num-windows", type=int, default=500, help="Number of windows to sample")
    parser.add_argument("--window-size", type=int, default=32, help="Window size in tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="../../data/slimpajama_train", help="Data directory")
    args = parser.parse_args()
    
    print("Loading embeddinggemma-300m...")
    model = SentenceTransformer("google/embeddinggemma-300m")
    tokenizer = model.tokenizer
    
    print(f"\nLoading {args.num_windows} diverse windows...")
    windows = load_diverse_windows(
        data_dir=args.data_dir,
        num_windows=args.num_windows,
        window_size=args.window_size,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    
    centroid_counts = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    distances = [1, 2, 4, 8, 16, 32]
    
    results = run_experiment(
        model=model,
        windows=windows,
        centroid_counts=centroid_counts,
        distances=distances,
        seed=args.seed,
    )
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Random pairs same-cluster rate")
    print("=" * 70)
    print(f"{'Centroids':>12} | {'Same Cluster':>12} | {'Unique Used':>12} | {'Coverage':>10}")
    print("-" * 55)
    for num_centroids in centroid_counts:
        r = results[num_centroids]
        coverage = r["unique_clusters"] / num_centroids
        print(f"{num_centroids:>12,} | {r['same_cluster_rate']:>11.2%} | {r['unique_clusters']:>12,} | {coverage:>9.1%}")


if __name__ == "__main__":
    main()

