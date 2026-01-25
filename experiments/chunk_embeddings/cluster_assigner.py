"""
ClusterAssigner: Assign cluster IDs to token sequences for chunk embeddings.

Handles windowing with correct causality: cluster_id at position p depends only
on tokens at positions 0..p-1 (the context used to predict token at position p).

Usage:
    assigner = ClusterAssigner("centroids_1m_w32_slimpajama.pt")
    cluster_ids = assigner.assign(token_ids)  # (batch, seq_len) -> (batch, seq_len)
"""

import hashlib
import os

import torch
import tiktoken

# Support imports from both project root and local directory
try:
    from experiments.chunk_embeddings.embedder import WindowEmbedder
    from experiments.chunk_embeddings.topk import MatryoshkaSegmentedTopKIndex
except ImportError:
    from embedder import WindowEmbedder
    from topk import MatryoshkaSegmentedTopKIndex


class ClusterAssigner:
    """
    Assigns cluster IDs to each position in token sequences.

    For position p, computes cluster_id from the causal context window:
    tokens[max(0, p - window_size) : p]  (excludes token at position p)

    This ensures no future information leakage for autoregressive models.
    """

    def __init__(
        self,
        centroid_path: str,
        window_size: int = 32,
        device: str = "cuda",
        compile_embedder: bool = True,
    ):
        """
        Args:
            centroid_path: Path to .pt file with centroid embeddings (K, 768)
            window_size: Context window size (default 32)
            device: Device for computation
            compile_embedder: Whether to use torch.compile for embedder
        """
        self.window_size = window_size
        self.device = device

        # Load centroids
        print(f"Loading centroids from {centroid_path}...")
        self.centroids = torch.load(centroid_path, weights_only=True)
        self.centroids = self.centroids.to(device=device, dtype=torch.float16)
        self.num_centroids = self.centroids.shape[0]
        print(
            f"  Loaded {self.num_centroids:,} centroids, shape {self.centroids.shape}"
        )

        # Build matryoshka segmented index for fast top-k search (~4x faster than brute force)
        print("  Building MatryoshkaSegmentedTopKIndex (fp8)...")
        self.index = MatryoshkaSegmentedTopKIndex(
            self.centroids, precision="fp8", device=device
        )
        print(
            f"  Index built: {self.index.n_segments} segments, "
            f"{self.index.segment_size} centroids/segment, "
            f"matryoshka_dim={self.index.matryoshka_dim}"
        )

        # Load embedder
        self.embedder = WindowEmbedder(
            compile=compile_embedder,
            max_seq_length=window_size,
            device=device,
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _get_causal_windows(
        self,
        token_ids: list[int],
        positions: list[int] | None = None,
    ) -> list[str]:
        """
        Extract causal context windows for specified positions.

        For position p, returns tokens[max(0, p - window_size) : p].
        Position 0 gets an empty string (no context).

        Args:
            token_ids: List of token IDs for a single sequence
            positions: Positions to extract windows for. If None, all positions.

        Returns:
            List of window text strings, one per position
        """
        if positions is None:
            positions = list(range(len(token_ids)))

        windows = []
        for p in positions:
            if p == 0:
                # No context for first position
                windows.append("")
            else:
                # Causal window: tokens before position p
                start = max(0, p - self.window_size)
                window_tokens = token_ids[start:p]  # Excludes position p
                window_text = self.tokenizer.decode(window_tokens)
                windows.append(window_text)

        return windows

    def assign_single(
        self,
        token_ids: list[int],
        stride: int = 1,
        return_positions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Assign cluster IDs to positions in a single sequence.

        Args:
            token_ids: Token IDs for one sequence
            stride: Assign every `stride` positions (1 = all positions)
            return_positions: If True, also return the positions that were assigned

        Returns:
            cluster_ids: Tensor of shape (num_positions,) with cluster IDs
            positions: (optional) Tensor of positions that were assigned
        """
        seq_len = len(token_ids)
        positions = list(range(0, seq_len, stride))

        # Get causal windows
        windows = self._get_causal_windows(token_ids, positions)

        # Handle empty windows (position 0 or short sequences)
        # For empty context, use cluster 0 as a sentinel
        non_empty_mask = [len(w) > 0 for w in windows]
        non_empty_windows = [w for w, m in zip(windows, non_empty_mask) if m]

        cluster_ids = torch.zeros(len(positions), dtype=torch.int32, device=self.device)

        if non_empty_windows:
            # Embed and assign using segmented index (~3.5x faster)
            embeddings = self.embedder.embed(non_empty_windows)  # (N, 768) fp16
            assigned, _ = self.index.search(embeddings, k=1)  # (N, 1)
            assigned = assigned.squeeze(1)  # (N,)

            # Fill in the non-empty positions
            j = 0
            for i, m in enumerate(non_empty_mask):
                if m:
                    cluster_ids[i] = assigned[j]
                    j += 1

        if return_positions:
            return cluster_ids, torch.tensor(positions, dtype=torch.int32)
        return cluster_ids

    def assign_batch(
        self,
        token_ids_batch: list[list[int]],
        stride: int = 1,
    ) -> list[torch.Tensor]:
        """
        Assign cluster IDs to a batch of sequences.

        Args:
            token_ids_batch: List of token ID lists
            stride: Assign every `stride` positions

        Returns:
            List of cluster_id tensors, one per sequence
        """
        # return_positions=False (default), so always returns Tensor
        results: list[torch.Tensor] = [
            self.assign_single(ids, stride=stride)  # type: ignore[misc]
            for ids in token_ids_batch
        ]
        return results

    def get_cluster_ids(
        self,
        x: torch.Tensor,
        stride: int = 1,
        cache_dir: str | None = None,
    ) -> torch.Tensor:
        """
        Compute or load cached cluster IDs for a batch of token IDs.

        Uses striding to reduce computation: computes cluster IDs every `stride`
        positions, then repeats each ID to fill all positions up to the next one.
        This is causal: each position gets a cluster_id computed from strictly
        earlier context.

        NOTE: This method uses .numpy().tobytes() and is NOT torch.compile compatible.
        Call it before any compiled forward pass.

        Args:
            x: Token IDs of shape (batch, seq_len)
            stride: Assign every `stride` positions (default 1 = all positions)
            cache_dir: Directory for disk cache. If None, no caching.

        Returns:
            cluster_ids: Tensor of shape (batch, seq_len) on same device as x
        """
        _, seq_len = x.shape

        # Compute fingerprint of batch (includes stride in case config changes)
        x_bytes = x.cpu().numpy().tobytes()
        fingerprint = hashlib.sha256(x_bytes + str(stride).encode()).hexdigest()[:16]

        # Check disk cache
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{fingerprint}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=True).to(x.device)

        # Compute cluster IDs with striding
        token_ids_list = [row.tolist() for row in x.cpu()]
        strided_ids_list = self.assign_batch(token_ids_list, stride=stride)

        # Expand strided cluster IDs to full sequence length
        # Each cluster ID at position i covers positions [i*stride, (i+1)*stride)
        cluster_ids_list = []
        for strided_ids in strided_ids_list:
            # strided_ids has shape (num_positions,) where num_positions = ceil(seq_len/stride)
            # Repeat each ID `stride` times, then truncate to seq_len
            expanded = strided_ids.repeat_interleave(stride)[:seq_len]
            cluster_ids_list.append(expanded)

        cluster_ids = torch.stack(cluster_ids_list).to(x.device)

        # Save to disk cache
        if cache_path is not None:
            torch.save(cluster_ids.cpu(), cache_path)

        return cluster_ids
