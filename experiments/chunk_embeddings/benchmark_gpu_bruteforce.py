"""GPU brute force top-k implementation with segmented search."""

import torch
import numpy as np
import time
import argparse
from embedder import WindowEmbedder
from sampling import sample_random_windows


def topk_segmented(
    queries: torch.Tensor,  # (num_queries, dim)
    centroids: torch.Tensor,  # (num_centroids, dim)
    k_segments: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find top-1 nearest neighbor using segmented search.

    1. Split centroids into k_segments
    2. Find top-1 in each segment
    3. Find overall top-1 from the k candidates

    Returns: (indices, similarities) both of shape (num_queries,)
    """
    num_centroids = centroids.shape[0]
    segment_size = (num_centroids + k_segments - 1) // k_segments

    # Collect top-1 from each segment
    top_indices = []
    top_sims = []

    for i in range(k_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, num_centroids)
        if start >= num_centroids:
            break

        segment = centroids[start:end]  # (segment_size, dim)
        sims = queries @ segment.T  # (num_queries, segment_size)

        seg_sims, seg_indices = sims.max(dim=1)  # (num_queries,)
        top_indices.append(seg_indices + start)  # Adjust to global index
        top_sims.append(seg_sims)

    # Stack candidates: (num_queries, k_segments)
    top_indices = torch.stack(top_indices, dim=1)
    top_sims = torch.stack(top_sims, dim=1)

    # Find best among candidates
    best_seg = top_sims.argmax(dim=1)  # (num_queries,)
    best_indices = top_indices[torch.arange(len(queries)), best_seg]
    best_sims = top_sims[torch.arange(len(queries)), best_seg]

    return best_indices, best_sims


def topk_bruteforce(
    queries: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact brute force top-1."""
    sims = queries @ centroids.T  # (num_queries, num_centroids)
    best_sims, best_indices = sims.max(dim=1)
    return best_indices, best_sims


def benchmark(num_centroids: int, num_queries: int, dim: int = 768):
    """Run benchmark."""
    print(f"\n{'='*60}")
    print(f"Scale: {num_centroids:,} centroids, {num_queries:,} queries, dim={dim}")
    print(
        f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate real Gemma embeddings
    print("\nLoading embedding model...")
    embedder = WindowEmbedder(compile=False)

    print(f"Sampling {num_centroids} centroid windows...")
    centroid_texts = sample_random_windows(
        "../../data/slimpajama_train", num_centroids, 32, embedder.tokenizer, seed=42
    )
    print(f"Sampling {num_queries} query windows...")
    query_texts = sample_random_windows(
        "../../data/slimpajama_train", num_queries, 32, embedder.tokenizer, seed=123
    )

    print("Embedding centroids...")
    centroids = embedder.embed(centroid_texts).float()
    print("Embedding queries...")
    queries = embedder.embed(query_texts).float()
    print(f"  Centroids: {centroids.shape}, {centroids.nbytes / 1e9:.3f} GB")
    print(f"  Queries: {queries.shape}, {queries.nbytes / 1e9:.3f} GB")

    # Warmup
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    _ = topk_bruteforce(queries[:100], centroids[:1000])
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Brute force FP32
    print("\n--- FP32 ---")
    print("Brute force (exact)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    indices_exact, sims_exact = topk_bruteforce(queries, centroids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    bf_time = time.time() - start
    print(f"  Time: {bf_time:.2f}s ({num_queries/bf_time:,.0f} q/s)")

    # # Segmented search FP32
    # for k_seg in [8]:
    #     print(f"Segmented (k={k_seg})...")
    #     torch.cuda.synchronize() if torch.cuda.is_available() else None
    #     start = time.time()
    #     indices_seg, sims_seg = topk_segmented(queries, centroids, k_segments=k_seg)
    #     torch.cuda.synchronize() if torch.cuda.is_available() else None
    #     seg_time = time.time() - start
    #     recall = (indices_seg == indices_exact).float().mean().item() * 100
    #     print(
    #         f"  Time: {seg_time:.2f}s ({num_queries/seg_time:,.0f} q/s), recall: {recall:.1f}%"
    #     )

    # FP16
    print("\n--- FP16 ---")
    centroids_fp16 = centroids.half()
    queries_fp16 = queries.half()
    print("Brute force...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    indices_fp16, _ = topk_bruteforce(queries_fp16, centroids_fp16)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fp16_time = time.time() - start
    recall = (indices_fp16 == indices_exact).float().mean().item() * 100
    print(
        f"  Time: {fp16_time:.2f}s ({num_queries/fp16_time:,.0f} q/s), recall: {recall:.1f}%"
    )

    # Matryoshka: first 256 dims
    print("\n--- Matryoshka (first 256 dims) ---")
    coarse_dim = 256
    centroids_coarse = centroids[:, :coarse_dim]
    centroids_coarse = centroids_coarse / centroids_coarse.norm(dim=1, keepdim=True)
    queries_coarse = queries[:, :coarse_dim]
    queries_coarse = queries_coarse / queries_coarse.norm(dim=1, keepdim=True)

    print("Brute force (coarse only)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    indices_coarse, _ = topk_bruteforce(queries_coarse, centroids_coarse)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    coarse_time = time.time() - start
    recall = (indices_coarse == indices_exact).float().mean().item() * 100
    print(
        f"  Time: {coarse_time:.2f}s ({num_queries/coarse_time:,.0f} q/s), recall: {recall:.1f}%"
    )

    # Two-stage: coarse to get top-k, then rerank with full (batched to avoid OOM)
    print("Two-stage (top-100 coarse, rerank full)...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    # Get top-100 from coarse
    sims_coarse = queries_coarse @ centroids_coarse.T
    _, topk_indices = sims_coarse.topk(100, dim=1)
    del sims_coarse  # Free memory
    # Rerank with full embeddings in batches
    batch_size = 10000
    indices_twostage = torch.empty(num_queries, dtype=torch.long, device=device)
    for i in range(0, num_queries, batch_size):
        end = min(i + batch_size, num_queries)
        batch_topk = topk_indices[i:end]  # (batch, 100)
        batch_queries = queries[i:end]  # (batch, dim)
        candidates = centroids[batch_topk]  # (batch, 100, dim)
        sims_full = torch.bmm(candidates, batch_queries.unsqueeze(2)).squeeze(
            2
        )  # (batch, 100)
        best_local = sims_full.argmax(dim=1)
        indices_twostage[i:end] = batch_topk[
            torch.arange(end - i, device=device), best_local
        ]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    twostage_time = time.time() - start
    recall = (indices_twostage == indices_exact).float().mean().item() * 100
    print(
        f"  Time: {twostage_time:.2f}s ({num_queries/twostage_time:,.0f} q/s), recall: {recall:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    benchmark(args.centroids, args.queries, args.dim)


if __name__ == "__main__":
    main()
