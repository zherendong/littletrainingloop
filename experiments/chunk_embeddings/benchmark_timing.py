"""Timing-only benchmark with synthetic data.

Compares:
- FP32 brute force
- FP16 brute force  
- Segmented search (k=8)
- Two-stage: 256-dim coarse → full rerank
"""

import torch
import time
import argparse


def benchmark(num_centroids: int, num_queries: int, dim: int = 768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nScale: {num_centroids:,} centroids, {num_queries:,} queries")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Synthetic normalized embeddings
    torch.manual_seed(42)
    centroids = torch.randn(num_centroids, dim, device=device)
    centroids = centroids / centroids.norm(dim=1, keepdim=True)
    queries = torch.randn(num_queries, dim, device=device)
    queries = queries / queries.norm(dim=1, keepdim=True)

    # Warmup
    _ = queries[:100] @ centroids[:1000].T
    torch.cuda.synchronize()

    # FP32 brute force
    torch.cuda.synchronize()
    t0 = time.time()
    idx_fp32 = (queries @ centroids.T).argmax(dim=1)
    torch.cuda.synchronize()
    t_fp32 = time.time() - t0
    print(f"FP32 brute:     {t_fp32*1000:8.2f} ms  ({num_queries/t_fp32:,.0f} q/s)")

    # FP16 brute force
    c16, q16 = centroids.half(), queries.half()
    torch.cuda.synchronize()
    t0 = time.time()
    idx_fp16 = (q16 @ c16.T).argmax(dim=1)
    torch.cuda.synchronize()
    t_fp16 = time.time() - t0
    recall = (idx_fp16 == idx_fp32).float().mean().item() * 100
    print(f"FP16 brute:     {t_fp16*1000:8.2f} ms  ({num_queries/t_fp16:,.0f} q/s)  recall: {recall:.1f}%")

    # Segmented k=8
    k_seg = 8
    seg_size = (num_centroids + k_seg - 1) // k_seg
    torch.cuda.synchronize()
    t0 = time.time()
    top_idx, top_sim = [], []
    for i in range(k_seg):
        start, end = i * seg_size, min((i + 1) * seg_size, num_centroids)
        sims = queries @ centroids[start:end].T
        seg_sim, seg_idx = sims.max(dim=1)
        top_idx.append(seg_idx + start)
        top_sim.append(seg_sim)
    top_idx = torch.stack(top_idx, dim=1)
    top_sim = torch.stack(top_sim, dim=1)
    best_seg = top_sim.argmax(dim=1)
    idx_seg = top_idx[torch.arange(num_queries, device=device), best_seg]
    torch.cuda.synchronize()
    t_seg = time.time() - t0
    recall = (idx_seg == idx_fp32).float().mean().item() * 100
    print(f"Segmented k=8:  {t_seg*1000:8.2f} ms  ({num_queries/t_seg:,.0f} q/s)  recall: {recall:.1f}%")

    # Two-stage: 256-dim coarse → rerank
    c256 = centroids[:, :256].contiguous()
    c256 = c256 / c256.norm(dim=1, keepdim=True)
    q256 = queries[:, :256].contiguous()
    q256 = q256 / q256.norm(dim=1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time.time()
    # Coarse: get top-100
    sims_coarse = q256 @ c256.T
    _, topk_idx = sims_coarse.topk(100, dim=1)
    # Rerank in batches
    batch_size = 10000
    idx_2stage = torch.empty(num_queries, dtype=torch.long, device=device)
    for i in range(0, num_queries, batch_size):
        end = min(i + batch_size, num_queries)
        batch_topk = topk_idx[i:end]
        batch_q = queries[i:end]
        candidates = centroids[batch_topk]
        sims_full = torch.bmm(candidates, batch_q.unsqueeze(2)).squeeze(2)
        best_local = sims_full.argmax(dim=1)
        idx_2stage[i:end] = batch_topk[torch.arange(end - i, device=device), best_local]
    torch.cuda.synchronize()
    t_2stage = time.time() - t0
    recall = (idx_2stage == idx_fp32).float().mean().item() * 100
    print(f"Two-stage:      {t_2stage*1000:8.2f} ms  ({num_queries/t_2stage:,.0f} q/s)  recall: {recall:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centroids", type=int, default=100000)
    parser.add_argument("--queries", type=int, default=10000)
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    benchmark(args.centroids, args.queries)


if __name__ == "__main__":
    main()

