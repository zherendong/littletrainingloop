"""Benchmark recall@k for different precision/dimension variants.

Measures how often the true top-1 (fp16 @ 768 dims) appears in the top-k
results for approximate methods.

Uses the actual TopKIndex and SegmentedTopKIndex classes from topk.py.

Variants tested:
- fp16 @ 768d (ground truth baseline)
- Segmented variants with matryoshka + reranking
"""

import torch
import time
import argparse
from topk import (
    TopKIndex,
    SegmentedTopKIndex,
    MatryoshkaSegmentedTopKIndex,
)


def load_centroids(path: str, device: str = "cuda") -> torch.Tensor:
    """Load pre-computed centroids."""
    print(f"Loading centroids from {path}...")
    centroids = torch.load(path, weights_only=True, map_location=device)
    print(f"  Shape: {centroids.shape}, dtype: {centroids.dtype}")
    return centroids


def load_queries(
    queries_path: str = "queries_10k_w32_slimpajama.pt",
    device: str = "cuda",
) -> torch.Tensor:
    """Load pre-computed query embeddings from disk."""
    print(f"\nLoading queries from {queries_path}...")
    queries = torch.load(queries_path, weights_only=True, map_location=device)
    print(f"  Query shape: {queries.shape}, dtype: {queries.dtype}")
    return queries


def truncate_and_normalize(embeddings: torch.Tensor, dim: int) -> torch.Tensor:
    """Truncate to first `dim` dimensions and re-normalize (matryoshka)."""
    truncated = embeddings[:, :dim].clone()
    truncated = truncated / truncated.norm(dim=1, keepdim=True)
    return truncated


def compute_recall_at_k(top1_gt: torch.Tensor, topk_approx: torch.Tensor) -> float:
    """Compute what fraction of ground truth top-1 appears in top-k."""
    matches = (topk_approx == top1_gt.unsqueeze(1)).any(dim=1)
    return matches.float().mean().item() * 100


def time_search(
    index: TopKIndex | SegmentedTopKIndex | MatryoshkaSegmentedTopKIndex,
    queries: torch.Tensor,
    k: int,
    n_runs: int = 5,
    name: str = "search",
) -> float:
    """Time the search operation only (index already built)."""
    # Warmup
    with torch.cuda.nvtx.range(f"{name}_warmup"):
        _ = index.search(queries[:100], k=k)
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.cuda.nvtx.range(f"{name}_run{i}"):
            _ = index.search(queries, k=k)
            torch.cuda.synchronize()
        times.append(time.time() - start)

    return min(times)  # Return best time


def benchmark(centroids_path: str, k: int = 16):
    """Run the full benchmark using TopKIndex from topk.py."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")

    # Load centroids
    centroids = load_centroids(centroids_path, device)
    num_centroids = centroids.shape[0]

    # Load pre-computed queries
    queries = load_queries(device=device)
    num_queries = queries.shape[0]

    # Build all indices upfront (not timed)
    print("\nBuilding indices...")
    idx_fp16_768 = TopKIndex(centroids, precision="fp16", device=device)
    # Segmented variant (no matryoshka, for comparison)
    idx_seg_fp16_768 = SegmentedTopKIndex(
        centroids, n_segments=256, precision="fp16", device=device
    )
    # Matryoshka + Segmented + Reranking variants (with new defaults: 256 segments, compile=True)
    idx_matr_256 = MatryoshkaSegmentedTopKIndex(
        centroids, n_segments=256, matryoshka_dim=256, device=device
    )
    idx_matr_256_fp8 = MatryoshkaSegmentedTopKIndex(
        centroids, n_segments=256, matryoshka_dim=256, precision="fp8", device=device
    )
    print(f"  Segmented: 256 segments, {idx_seg_fp16_768.segment_size} centroids each")
    print("  Done.")

    print(f"\n{'='*70}")
    print(f"Benchmark: {num_centroids:,} centroids, {num_queries:,} queries, k={k}")
    print(f"{'='*70}")

    results = {}

    # Ground truth: fp16 @ 768d (exact), top-1
    print("\n[Baseline] fp16 @ 768d (exact)...")
    with torch.cuda.nvtx.range("exact_fp16_768d_groundtruth"):
        top1_gt, _ = idx_fp16_768.search(queries, k=1)
        top1_gt = top1_gt.squeeze(1)
    gt_time = time_search(idx_fp16_768, queries, k=k, name="exact_fp16_768d")
    print(f"  Time: {gt_time:.3f}s ({num_queries/gt_time:,.0f} q/s)")
    results["exact_fp16_768d"] = {"time": gt_time, "recall": 100.0}

    # Segmented (no matryoshka, for comparison)
    print("\n[seg_768d] Segmented @ 768d (no reranking)...")
    t = time_search(idx_seg_fp16_768, queries, k=k, name="seg_768d")
    topk_result, _ = idx_seg_fp16_768.search(queries, k=k)
    recall = compute_recall_at_k(top1_gt, topk_result)
    print(f"  Time: {t:.3f}s ({num_queries/t:,.0f} q/s)")
    print(f"  Recall@{k}: {recall:.2f}%")
    results["seg_768d"] = {"time": t, "recall": recall}

    # Matryoshka + Segmented + Reranking variants
    matr_variants = [
        ("matr_256d", idx_matr_256),
        ("matr_256d_fp8", idx_matr_256_fp8),
    ]

    for variant_name, idx in matr_variants:
        print(f"\n[{variant_name}] Matryoshka + Segmented + Rerank...")
        t = time_search(idx, queries, k=k, name=variant_name)
        topk_result, _ = idx.search(queries, k=k)
        recall = compute_recall_at_k(top1_gt, topk_result)
        print(f"  Time: {t:.3f}s ({num_queries/t:,.0f} q/s)")
        print(f"  Recall@{k}: {recall:.2f}%")
        results[variant_name] = {"time": t, "recall": recall}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Time (s)':<12} {'Speedup':<10} {'Recall@'+str(k):<12}")
    print("-" * 70)
    for method, data in results.items():
        speedup = gt_time / data["time"]
        print(
            f"{method:<20} {data['time']:<12.3f} {speedup:<10.2f}x {data['recall']:<12.2f}%"
        )

    return results, top1_gt, queries, centroids


def benchmark_recall_curve(
    top1_gt: torch.Tensor,
    queries: torch.Tensor,
    centroids: torch.Tensor,
):
    """Run benchmark for multiple k values to see recall curve.

    Re-uses data from main benchmark to avoid re-embedding.
    """
    device = "cuda"

    # Build indices
    idx_seg_768 = SegmentedTopKIndex(
        centroids, n_segments=1024, precision="fp16", device=device
    )
    idx_matr_256 = MatryoshkaSegmentedTopKIndex(
        centroids, n_segments=1024, matryoshka_dim=256, precision="fp16", device=device
    )
    idx_matr_256_fp8 = MatryoshkaSegmentedTopKIndex(
        centroids, n_segments=1024, matryoshka_dim=256, precision="fp8", device=device
    )

    print(f"\n{'='*70}")
    print("RECALL CURVE: Recall@k for different k values")
    print(f"{'='*70}")

    k_values = [16]

    print(f"\n{'k':<6} {'seg_768':<12} {'matr_256':<12} {'matr_256_fp8':<12}")
    print("-" * 50)

    results_curve = {
        "k": k_values,
        "seg_768": [],
        "matr_256": [],
        "matr_256_fp8": [],
    }

    for k in k_values:
        topk_1, _ = idx_seg_768.search(queries, k=k)
        recall_1 = compute_recall_at_k(top1_gt, topk_1)

        topk_2, _ = idx_matr_256.search(queries, k=k)
        recall_2 = compute_recall_at_k(top1_gt, topk_2)

        topk_3, _ = idx_matr_256_fp8.search(queries, k=k)
        recall_3 = compute_recall_at_k(top1_gt, topk_3)

        results_curve["seg_768"].append(recall_1)
        results_curve["matr_256"].append(recall_2)
        results_curve["matr_256_fp8"].append(recall_3)

        print(f"{k:<6} {recall_1:<12.2f}% {recall_2:<12.2f}% {recall_3:<12.2f}%")

    return results_curve


def benchmark_matryoshka_dims(
    top1_gt: torch.Tensor,
    queries: torch.Tensor,
    centroids: torch.Tensor,
    k: int = 16,
):
    """Benchmark matryoshka + segmented + reranking at different dimensions.

    Re-uses data from main benchmark.
    """
    device = "cuda"

    dims = [256, 512, 768]

    print(f"\n{'Dims':<8} {'Recall@'+str(k):<14} {'Time (s)':<12} {'Speedup':<12}")
    print("-" * 50)

    # Baseline timing with exact search
    idx_baseline = TopKIndex(centroids, precision="fp16", device=device)
    baseline_time = time_search(idx_baseline, queries, k=k)

    for dim in dims:
        # Build matryoshka + segmented + reranking index
        idx = MatryoshkaSegmentedTopKIndex(
            centroids,
            n_segments=1024,
            matryoshka_dim=dim,
            precision="fp16",
            device=device,
        )

        # Time and get results
        dim_time = time_search(idx, queries, k=k)
        topk_result, _ = idx.search(queries, k=k)

        recall = compute_recall_at_k(top1_gt, topk_result)
        speedup = baseline_time / dim_time

        print(f"{dim:<8} {recall:<14.2f}% {dim_time:<12.3f} {speedup:<12.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--centroids", type=str, default="centroids_1m_w32_slimpajama.pt"
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--curve",
        action="store_true",
        help="Also run recall curve for multiple k values",
    )
    args = parser.parse_args()

    results, top1_gt, queries, centroids = benchmark(args.centroids, args.k)

    if args.curve:
        print("\n" + "=" * 70)
        print("Running recall curve benchmark...")
        curve_results = benchmark_recall_curve(top1_gt, queries, centroids)
