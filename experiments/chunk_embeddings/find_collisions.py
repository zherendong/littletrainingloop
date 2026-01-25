"""Find windows from different files that map to the same centroid.

This helps understand what kinds of text patterns cluster together.
"""

import argparse
import torch
from collections import defaultdict
from embedder import WindowEmbedder
from sampling import sample_random_windows
from topk import TopKIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument(
        "--centroids", type=str, default="centroids_100k_w32_slimpajama.pt"
    )
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    # Load centroids
    print(f"Loading centroids from {args.centroids}...")
    centroids = torch.load(args.centroids)
    index = TopKIndex(centroids, precision="fp16")

    # Load model
    print("Loading embedding model...")
    embedder = WindowEmbedder(compile=False)

    # Sample windows - use different seed to avoid overlap with centroids
    print(f"Sampling {args.num_queries} windows...")
    windows = sample_random_windows(
        "../../data/slimpajama_train",
        args.num_queries,
        window_size=32,
        tokenizer=embedder.tokenizer,
        seed=12345,
    )

    # Embed
    print("Embedding windows...")
    queries = embedder.embed(windows)

    # Search
    print("Finding nearest centroids...")
    indices, similarities = index.search(queries, k=1)
    indices = indices.squeeze().cpu().tolist()
    similarities = similarities.squeeze().cpu().tolist()

    # Group by centroid
    centroid_to_windows = defaultdict(list)
    for i, (centroid_id, sim) in enumerate(zip(indices, similarities)):
        centroid_to_windows[centroid_id].append((i, windows[i], sim))

    # Find collisions (centroids with multiple windows)
    collisions = {k: v for k, v in centroid_to_windows.items() if len(v) > 1}

    print(f"\n{'='*60}")
    print(
        f"Results: {len(windows)} windows → {len(centroid_to_windows)} unique centroids"
    )
    print(f"Collisions: {len(collisions)} centroids have multiple windows")
    print(f"{'='*60}")

    # Print examples
    examples_shown = 0
    for centroid_id, window_list in sorted(
        collisions.items(), key=lambda x: -len(x[1])
    ):
        if examples_shown >= args.max_examples:
            break

        print(f"\n--- Centroid {centroid_id} ({len(window_list)} windows) ---")
        for idx, window_text, sim in window_list[:3]:  # Show up to 3 per centroid
            # Truncate for display
            display_text = window_text[:200].replace("\n", "\\n")
            if len(window_text) > 200:
                display_text += "..."
            print(f"  [{idx}] (sim={sim:.3f}) {display_text}")

        examples_shown += 1


if __name__ == "__main__":
    main()
