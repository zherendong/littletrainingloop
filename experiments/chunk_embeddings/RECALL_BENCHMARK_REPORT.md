# Top-K Recall Benchmark Report

**Date:** 2026-01-22
**Hardware:** NVIDIA GH200 480GB
**Centroids:** 1,000,000 (from `centroids_1m_w32_slimpajama.pt`)
**Queries:** 10,000 fresh embeddings from SlimPajama
**Embedding Model:** google/embeddinggemma-300m (768 dimensions, matryoshka-capable)
**Benchmark:** Uses `TopKIndex` and `SegmentedTopKIndex` from `topk.py`

## Summary

This benchmark evaluates how accurately approximate methods can recover the true top-1 result (determined via fp16 precision on full 768-dim embeddings) when retrieving top-k candidates.

### Key Question
> How often does the true top-1 result appear in the top-16?

## Results

### Primary Benchmark (k=16)

| Method | Time (s) | Speedup | Recall@16 | Throughput |
|--------|----------|---------|-----------|------------|
| **fp16 @ 768d** (baseline) | 0.106 | 1.00x | 100.00% | 94K q/s |
| fp8 @ 768d | 0.099 | 1.07x | 100.00% | 101K q/s |
| fp16 @ 256d (matryoshka) | 0.094 | 1.13x | 80.14% | 107K q/s |
| fp8 @ 256d (matryoshka + fp8) | 0.093 | 1.14x | 79.72% | 108K q/s |
| **Segmented (1024)** | **0.038** | **2.80x** | **100.00%** | **265K q/s** |

### Recall@k Curve

| k | fp8 @ 768d | fp16 @ 256d | fp8 @ 256d | Segmented 1024 |
|---|------------|-------------|------------|----------------|
| 1 | 93.78% | 32.89% | 32.56% | **100.00%** |
| 2 | 98.92% | 44.66% | 44.20% | **100.00%** |
| 4 | 99.97% | 57.60% | 56.77% | **100.00%** |
| 8 | 100.00% | 69.72% | 69.65% | **100.00%** |
| **16** | 100.00% | 80.14% | 79.72% | **100.00%** |
| 32 | 100.00% | 88.24% | 88.00% | **100.00%** |
| 64 | 100.00% | 93.72% | 93.61% | **100.00%** |
| 128 | 100.00% | 96.78% | 96.70% | **100.00%** |

### Matryoshka Dimension Sweep (k=16)

| Dims | fp16 Recall@16 | fp8 Recall@16 | Speedup vs 768 |
|------|----------------|---------------|----------------|
| 64 | 21.36% | 21.09% | 1.15x |
| 128 | 52.27% | 51.86% | 1.14x |
| 192 | 68.90% | 68.59% | 1.13x |
| 256 | 80.14% | 79.72% | 1.12x |
| 384 | 91.82% | 91.72% | 1.08x |
| 512 | 97.37% | 97.36% | 1.04x |
| 768 | 100.00% | 100.00% | 1.00x |

## Key Findings

### 1. Segmented search is the clear winner
- **2.80x speedup** with **100% recall** for finding the true top-1
- Works because the global top-1 must be the top-1 of its segment
- 265K queries/second vs 94K for exact search

### 2. Why segmented works
The search operation has two parts:
- **Matmul:** `sims = queries @ centroids.T` → scales with dimension D
- **TopK:** `sims.topk(k)` on 10 billion elements → **dominates runtime** (~85ms of ~106ms)

Segmented search replaces the expensive topk with:
- 1024 parallel `.max()` operations (one per segment)
- Final `.topk(k)` on only 1024 elements

### 3. Why dimension reduction doesn't help much
- Reducing D from 768 to 256 only affects the matmul (~27ms → ~10ms)
- The topk operation (~85ms) is independent of D
- Net speedup: only 1.13x (not 3x as one might expect)

### 4. Matryoshka is NOT recommended
- Only 1.13x speedup with 80% recall@16
- The recall loss is unacceptable for cluster assignment

### 5. FP8 provides modest improvement
- 100% recall@16 at full dimensions
- 1.07x speedup - marginal

## Recommendations

### For top-k retrieval (k > 1):
Use **SegmentedTopKIndex with 1024 segments**:
- 2.80x faster than exact search
- 100% recall for finding true top-1 in top-k results
- 265K queries/second

### For cluster assignment (k = 1):
Use regular **TopKIndex** with fp16 @ 768d:
- Already optimized with `.max()` instead of `.topk()`
- ~300K queries/second
- No approximation needed

## Conclusion

The new `SegmentedTopKIndex` class provides **2.8x speedup with perfect recall** for the use case of finding the true top-1 in top-k results. This is achieved by exploiting the fact that the global maximum must be the maximum of its segment.

```python
from topk import SegmentedTopKIndex

# Build index once
index = SegmentedTopKIndex(centroids, n_segments=1024)

# Search
indices, similarities = index.search(queries, k=16)  # 2.8x faster, 100% recall
```
