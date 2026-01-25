"""Generate centroid embeddings from random SlimPajama windows.

Usage:
    python generate_centroids.py --num-centroids 100000 --output centroids.pt

Estimated time: ~15 min for 100K centroids on a single GPU.
"""

import argparse
import time
import torch
from embedder import WindowEmbedder
from sampling import sample_random_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-centroids", type=int, default=100_000)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default="../../data/slimpajama_train")
    parser.add_argument("--output", type=str, default="centroids.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.num_centroids:,} centroids")
    print(f"Window size: {args.window_size} tokens")
    print(f"Output: {args.output}")

    # Estimate time
    embed_rate = 8000  # ~8K embeddings/sec with compiled model
    est_time = args.num_centroids / embed_rate / 60
    print(f"Estimated time: {est_time:.1f} minutes")

    # Load model
    print("\nLoading embedding model...")
    t0 = time.time()
    embedder = WindowEmbedder(compile=True, batch_size=args.batch_size)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Sample windows
    print(f"\nSampling {args.num_centroids:,} windows...")
    t0 = time.time()
    windows = sample_random_windows(
        args.data_dir,
        args.num_centroids,
        args.window_size,
        embedder.tokenizer,
        seed=args.seed,
    )
    print(f"Sampled in {time.time() - t0:.1f}s")

    # Embed windows
    print(f"\nEmbedding {len(windows):,} windows (batch_size={args.batch_size})...")
    t0 = time.time()
    centroids = embedder.embed(windows)
    embed_time = time.time() - t0
    print(
        f"Embedded in {embed_time:.1f}s ({len(windows) / embed_time:.1f} windows/sec)"
    )
    print(f"\nCentroids shape: {centroids.shape}")
    print(f"Storage: {centroids.nbytes / 1e6:.1f} MB")

    # Save
    print(f"\nSaving to {args.output}...")
    torch.save(centroids, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
