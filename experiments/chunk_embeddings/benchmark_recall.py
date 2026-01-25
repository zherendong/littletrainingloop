"""Recall benchmark with real Gemma embeddings.

Compares FP32, FP16, FP8 precision and matryoshka dimensions.
"""

import torch
import time
from embedder import WindowEmbedder
from sampling import sample_random_windows


def benchmark(num_centroids: int, num_queries: int):
    device = "cuda"
    print(f"\nScale: {num_centroids:,} centroids, {num_queries:,} queries")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Load real embeddings
    print("\nLoading model...")
    embedder = WindowEmbedder(compile=False)

    print(f"Sampling windows...")
    centroid_texts = sample_random_windows(
        "../../data/slimpajama_train", num_centroids, 32, embedder.tokenizer, seed=42
    )
    query_texts = sample_random_windows(
        "../../data/slimpajama_train", num_queries, 32, embedder.tokenizer, seed=123
    )

    print("Embedding...")
    centroids = embedder.embed(centroid_texts).float()
    queries = embedder.embed(query_texts).float()
    print(f"Centroids: {centroids.shape}, Queries: {queries.shape}")

    # Ground truth: FP32
    idx_fp32 = (queries @ centroids.T).argmax(dim=1)

    print("\n--- Precision comparison ---")

    # FP16
    c16, q16 = centroids.half(), queries.half()
    idx_fp16 = (q16 @ c16.T).argmax(dim=1)
    recall = (idx_fp16 == idx_fp32).float().mean().item() * 100
    print(f"FP16:  recall {recall:.1f}%")

    # FP8 e4m3 native (using _scaled_mm)
    # Pad centroids to be divisible by 16 if needed
    n_centroids = centroids.shape[0]
    pad_n = (16 - n_centroids % 16) % 16
    if pad_n > 0:
        centroids_padded = torch.cat(
            [centroids, torch.zeros(pad_n, 768, device=device)], dim=0
        )
    else:
        centroids_padded = centroids

    c8 = centroids_padded.to(torch.float8_e4m3fn)
    q8 = queries.to(torch.float8_e4m3fn)
    scale_q = torch.ones(1, device=device, dtype=torch.float32)
    scale_c = torch.ones(1, device=device, dtype=torch.float32)
    # _scaled_mm needs column-major second arg: centroids.T viewed as column-major
    sims_fp8 = torch._scaled_mm(
        q8, c8.T, scale_a=scale_q, scale_b=scale_c, out_dtype=torch.float16
    )
    sims_fp8 = sims_fp8[:, :n_centroids]  # Remove padding
    idx_fp8 = sims_fp8.argmax(dim=1)
    recall = (idx_fp8 == idx_fp32).float().mean().item() * 100
    print(f"FP8 e4m3 (native): recall {recall:.1f}%")

    print("\n--- Matryoshka dimensions ---")
    for dim in [64, 128, 256, 384, 512]:
        c_trunc = centroids[:, :dim]
        c_trunc = c_trunc / c_trunc.norm(dim=1, keepdim=True)
        q_trunc = queries[:, :dim]
        q_trunc = q_trunc / q_trunc.norm(dim=1, keepdim=True)
        idx_trunc = (q_trunc @ c_trunc.T).argmax(dim=1)
        recall = (idx_trunc == idx_fp32).float().mean().item() * 100
        print(f"{dim:3d}-dim: recall {recall:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.centroids, args.queries)
