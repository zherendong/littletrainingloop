"""GPU brute force top-k nearest neighbor search.

Usage:
    from topk import TopKIndex

    index = TopKIndex(centroids)  # (N, D) tensor
    indices, similarities = index.search(queries, k=1)  # (Q, k) tensors
"""

import torch
import numpy as np


class TopKIndex:
    """Brute force top-k index with FP16 (default) or FP8 precision."""

    def __init__(
        self,
        centroids: torch.Tensor,
        precision: str = "fp16",  # "fp16" or "fp8"
        device: str = "cuda",
    ):
        """
        Args:
            centroids: (N, D) tensor of normalized embeddings
            precision: "fp16" (default, 99.7% recall) or "fp8" (96.9% recall)
            device: "cuda" or "cpu"
        """
        self.device = device
        self.precision = precision
        self.n_centroids = centroids.shape[0]
        self.dim = centroids.shape[1]

        if precision == "fp16":
            self.centroids = centroids.to(device=device, dtype=torch.float16)
        elif precision == "fp8":
            # Pad to multiple of 16 for _scaled_mm
            pad_n = (16 - self.n_centroids % 16) % 16
            if pad_n > 0:
                centroids_padded = torch.cat(
                    [centroids, torch.zeros(pad_n, self.dim, device=device)], dim=0
                )
            else:
                centroids_padded = centroids.to(device=device)
            self.centroids = centroids_padded.to(torch.float8_e4m3fn)
            self.scale = torch.ones(1, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown precision: {precision}")

    def search(
        self, queries: torch.Tensor, k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find top-k nearest centroids for each query.

        Args:
            queries: (Q, D) tensor of normalized query embeddings
            k: number of nearest neighbors to return

        Returns:
            indices: (Q, k) tensor of centroid indices
            similarities: (Q, k) tensor of cosine similarities
        """
        if self.precision == "fp16":
            q = queries.to(device=self.device, dtype=torch.float16)
            sims = q @ self.centroids.T
        else:  # fp8
            q = queries.to(device=self.device, dtype=torch.float8_e4m3fn)
            sims = torch._scaled_mm(
                q,
                self.centroids.T,
                scale_a=self.scale,
                scale_b=self.scale,
                out_dtype=torch.float16,
            )
            sims = sims[:, : self.n_centroids]  # Remove padding

        if k == 1:
            similarities, indices = sims.max(dim=1, keepdim=True)
        else:
            similarities, indices = sims.topk(k, dim=1)

        return indices, similarities


class SegmentedTopKIndex:
    """Approximate top-k using segmented search.

    Instead of doing topk on N centroids (expensive), we:
    1. Segment centroids into S segments of ~N/S each
    2. Find top-1 from each segment (cheap - just argmax)
    3. Do final topk on S candidates

    This reduces topk cost from O(N) to O(S) where S << N.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        n_segments: int = 1024,
        precision: str = "fp16",
        device: str = "cuda",
    ):
        """
        Args:
            centroids: (N, D) tensor of normalized embeddings
            n_segments: number of segments (default 1024)
            precision: "fp16" or "fp8"
            device: "cuda" or "cpu"
        """
        self.device = device
        self.precision = precision
        self.n_centroids = centroids.shape[0]
        self.dim = centroids.shape[1]
        self.n_segments = n_segments

        # Compute segment size and pad if needed
        self.segment_size = (self.n_centroids + n_segments - 1) // n_segments
        self.padded_size = self.segment_size * n_segments

        # Pad centroids to exact multiple of n_segments
        if self.padded_size > self.n_centroids:
            padding = torch.zeros(
                self.padded_size - self.n_centroids,
                self.dim,
                device=device,
                dtype=centroids.dtype,
            )
            centroids_padded = torch.cat([centroids.to(device), padding], dim=0)
        else:
            centroids_padded = centroids.to(device)

        # Store as (n_segments, segment_size, dim) for efficient access
        if precision == "fp16":
            self.centroids = centroids_padded.to(dtype=torch.float16)
        elif precision == "fp8":
            self.centroids = centroids_padded.to(dtype=torch.float8_e4m3fn)
            self.scale = torch.ones(1, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown precision: {precision}")

        # Keep flat for matmul, reshape during search
        # self.centroids shape: (padded_size, dim)

    def search(
        self, queries: torch.Tensor, k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find approximate top-k using segmented search.

        Args:
            queries: (Q, D) tensor of normalized query embeddings
            k: number of nearest neighbors to return (must be <= n_segments)

        Returns:
            indices: (Q, k) tensor of centroid indices
            similarities: (Q, k) tensor of cosine similarities
        """
        assert k <= self.n_segments, f"k={k} must be <= n_segments={self.n_segments}"

        Q = queries.shape[0]

        # Compute all similarities: (Q, padded_size)
        with torch.cuda.nvtx.range("seg_matmul"):
            if self.precision == "fp16":
                q = queries.to(device=self.device, dtype=torch.float16)
                sims = q @ self.centroids.T
            else:  # fp8
                q = queries.to(device=self.device, dtype=torch.float8_e4m3fn)
                sims = torch._scaled_mm(
                    q,
                    self.centroids.T,
                    scale_a=self.scale,
                    scale_b=self.scale,
                    out_dtype=torch.float16,
                )

        with torch.cuda.nvtx.range("seg_segmented_max"):
            # Reshape to (Q, n_segments, segment_size)
            sims_segmented = sims.view(Q, self.n_segments, self.segment_size)

            # Find top-1 within each segment: (Q, n_segments)
            segment_max_sims, segment_local_idx = sims_segmented.max(dim=2)

            # Convert local indices to global indices
            segment_offsets = torch.arange(
                0, self.padded_size, self.segment_size, device=self.device
            )  # (n_segments,)
            segment_global_idx = segment_local_idx + segment_offsets  # (Q, n_segments)

        with torch.cuda.nvtx.range("seg_final_topk"):
            # Final topk over segment winners
            if k == 1:
                final_sims, winner_idx = segment_max_sims.max(dim=1, keepdim=True)
            else:
                final_sims, winner_idx = segment_max_sims.topk(k, dim=1)

            # Gather the global indices
            final_indices = torch.gather(segment_global_idx, 1, winner_idx)

            # Clamp to valid range (in case padding gave false positives)
            final_indices = final_indices.clamp(max=self.n_centroids - 1)

        return final_indices, final_sims


class MatryoshkaSegmentedTopKIndex:
    """Two-stage approximate top-k using matryoshka + segmented search + reranking.

    Stage 1: Use truncated (matryoshka) dimensions for fast segmented search
             to find top candidates from each segment.
    Stage 2: Rerank those candidates using full dimensions for final top-k.

    This combines the benefits of:
    - Matryoshka: faster matmul with lower dimensions
    - Segmented: faster topk with segment-wise max
    - Reranking: accurate final selection with full dimensions

    With 256 segments and matryoshka_dim=768, achieves 100% recall@16 while
    being ~1.8x faster than plain segmented search (with torch.compile).
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        n_segments: int = 1024,
        matryoshka_dim: int = 256,
        precision: str = "fp16",
        device: str = "cuda",
        compile: bool = True,
    ):
        """
        Args:
            centroids: (N, D) tensor of normalized embeddings (full dimensions)
            n_segments: number of segments (default 1024)
            matryoshka_dim: dimension to use for first-stage search (default 256)
            precision: "fp16" or "fp8" for first-stage search
            device: "cuda" or "cpu"
            compile: whether to use torch.compile for faster search (default True)
        """
        self.device = device
        self.precision = precision
        self.n_centroids = centroids.shape[0]
        self.full_dim = centroids.shape[1]
        self.matryoshka_dim = matryoshka_dim
        self.n_segments = n_segments

        # Store full-dim centroids for reranking
        if precision == "fp8":
            self.centroids_full = centroids.to(device=device, dtype=torch.float8_e4m3fn)
        else:
            self.centroids_full = centroids.to(device=device, dtype=torch.float16)

        # Truncate and renormalize for matryoshka search
        centroids_trunc = centroids[:, :matryoshka_dim]
        centroids_trunc = centroids_trunc / centroids_trunc.norm(dim=1, keepdim=True)

        # Compute segment size and pad if needed
        self.segment_size = (self.n_centroids + n_segments - 1) // n_segments
        self.padded_size = self.segment_size * n_segments

        # Precompute segment offsets
        self.segment_offsets = torch.arange(
            0, self.padded_size, self.segment_size, device=device
        )

        # Pad truncated centroids to exact multiple of n_segments
        if self.padded_size > self.n_centroids:
            padding = torch.zeros(
                self.padded_size - self.n_centroids,
                matryoshka_dim,
                device=device,
                dtype=centroids_trunc.dtype,
            )
            centroids_padded = torch.cat([centroids_trunc.to(device), padding], dim=0)
        else:
            centroids_padded = centroids_trunc.to(device)

        # Store truncated centroids for first-stage search
        if precision == "fp16":
            self.centroids_trunc = centroids_padded.to(dtype=torch.float16)
        elif precision == "fp8":
            self.centroids_trunc = centroids_padded.to(dtype=torch.float8_e4m3fn)
            self.scale = torch.ones(1, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown precision: {precision}")

        # Compile the search kernel
        self._compiled = compile
        if compile:
            self._search_impl = torch.compile(self._search_kernel, mode="max-autotune")
        else:
            self._search_impl = self._search_kernel

    def _search_kernel(
        self,
        queries: torch.Tensor,
        centroids_trunc: torch.Tensor,
        centroids_full: torch.Tensor,
        segment_offsets: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core search logic, separated for torch.compile."""
        Q = queries.shape[0]

        orig_queries = queries

        # Stage 1: Segmented search with truncated dimensions
        queries_trunc = queries[:, : self.matryoshka_dim]
        queries_trunc = queries_trunc / queries_trunc.norm(dim=1, keepdim=True)
        if self.precision == "fp8":
            queries_trunc = queries_trunc.to(dtype=torch.float8_e4m3fn)
            sims = torch._scaled_mm(
                queries_trunc,
                centroids_trunc.T,
                scale_a=self.scale,
                scale_b=self.scale,
                out_dtype=torch.float16,
            )
        else:
            queries_trunc = queries_trunc.to(dtype=torch.float16)
            sims = queries_trunc @ centroids_trunc.T

        sims_segmented = sims.view(Q, self.n_segments, self.segment_size)
        _, segment_local_idx = sims_segmented.max(dim=2)
        candidate_indices = segment_local_idx + segment_offsets
        candidate_indices = candidate_indices.clamp(max=self.n_centroids - 1)

        # Stage 2: Rerank candidates with full dimensions
        candidate_centroids = centroids_full[candidate_indices]  # gather in fp8 or fp16
        if self.precision == "fp8":
            # Convert back to fp16 for bmm (gathered in fp8 to save bandwidth)
            candidate_centroids = candidate_centroids.to(dtype=torch.float16)

        queries_full = orig_queries.to(dtype=torch.float16)
        rerank_sims = torch.bmm(
            queries_full.unsqueeze(1), candidate_centroids.transpose(1, 2)
        ).squeeze(1)

        # Final topk
        if k == 1:
            final_sims, winner_idx = rerank_sims.max(dim=1, keepdim=True)
        else:
            final_sims, winner_idx = rerank_sims.topk(k, dim=1, sorted=False)

        final_indices = torch.gather(candidate_indices, 1, winner_idx)
        return final_indices, final_sims

    def search(
        self, queries: torch.Tensor, k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find approximate top-k using two-stage search.

        Args:
            queries: (Q, D) tensor of normalized query embeddings (full dimensions)
            k: number of nearest neighbors to return (must be <= n_segments)

        Returns:
            indices: (Q, k) tensor of centroid indices
            similarities: (Q, k) tensor of cosine similarities (from reranking)
        """
        assert k <= self.n_segments, f"k={k} must be <= n_segments={self.n_segments}"

        queries = queries.to(device=self.device)
        return self._search_impl(
            queries,
            self.centroids_trunc,
            self.centroids_full,
            self.segment_offsets,
            k,
        )


def topk_search(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    k: int = 1,
    precision: str = "fp16",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Functional API for one-off searches.

    Args:
        queries: (Q, D) normalized query embeddings
        centroids: (N, D) normalized centroid embeddings
        k: number of nearest neighbors
        precision: "fp16" or "fp8"

    Returns:
        indices: (Q, k) centroid indices
        similarities: (Q, k) cosine similarities
    """
    index = TopKIndex(centroids, precision=precision)
    return index.search(queries, k=k)


def sample_query_positions(
    seq_len: int,
    window_size: int = 32,
    avg_stride: int = 8,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample positions for embedding queries with random stride.

    Uses geometric distribution so that on average we query every `avg_stride` tokens.
    Each position represents the END of a window (so position must be >= window_size).

    Args:
        seq_len: total sequence length
        window_size: context window size (default 32)
        avg_stride: average tokens between queries (default 8)
        seed: optional random seed for reproducibility

    Returns:
        positions: 1D tensor of positions to query (sorted)
    """

    # Start after first full window
    min_pos = window_size
    if seq_len <= min_pos:
        return torch.tensor([], dtype=torch.long)

    rng = np.random.default_rng(seed)

    # Sample gaps from geometric distribution
    # Geometric gives number of failures before first success
    # We want gaps with mean = avg_stride, so p = 1/avg_stride
    p = 1.0 / avg_stride

    # Estimate how many positions we need (with buffer)
    expected_count = (seq_len - min_pos) // avg_stride + 10
    gaps = rng.geometric(p, size=expected_count)
    gaps = np.maximum(gaps, 1)  # minimum gap of 1

    # Compute cumulative positions
    positions = min_pos + np.cumsum(gaps) - gaps[0]  # start at min_pos
    positions = positions[positions < seq_len]

    return torch.tensor(positions, dtype=torch.long)
