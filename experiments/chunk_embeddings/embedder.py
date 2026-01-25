"""
Window embedding using SentenceTransformer (embeddinggemma-300m).

Provides:
- WindowEmbedder: High-performance wrapper with torch.compile optimization
"""

import os
import torch
from sentence_transformers import SentenceTransformer

# Enable torch.compile cache for faster startup
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")


class WindowEmbedder:
    """
    Wrapper around SentenceTransformer for embedding text windows.

    Handles tokenization, left-truncation (keeps end of window), padding to
    fixed shapes, and batched inference. Optimized for torch.compile with
    consistent tensor shapes.

    Performance notes (GH200, compiled model with TF32, max_seq_length=32):
        - First batch: ~11s (autotuning, cached for subsequent runs)
        - Steady-state: ~50ms for 512 windows (10K windows/sec)
        - With stride=8: 100K tokens → 12.5K windows → ~1.2s embed time

    Args:
        compile: Whether to compile with torch.compile (max-autotune mode).
        use_tf32: Whether to enable TF32 for faster matmuls.
        max_seq_length: Fixed sequence length. Truncate from start, pad to this.
        batch_size: Batch size for encoding. Use 512 for best GPU utilization.
        device: Device for inference ("cuda" or "cpu").
    """

    def __init__(
        self,
        compile: bool = True,
        use_tf32: bool = False,
        max_seq_length: int = 32,
        batch_size: int = 512,
        device: str = "cuda",
    ):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

        print("Loading embeddinggemma-300m...")

        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  TF32 enabled")

        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.tokenizer = self.model.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id or 0

        if compile:
            print("  Compiling with torch.compile (max-autotune)...")
            self.model._first_module().auto_model = torch.compile(
                self.model._first_module().auto_model,
                mode="max-autotune",
            )
            print("  Model compiled (first inference will trigger autotuning)")

    def tokenize(self, windows: list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize windows with left-truncation and fixed-length padding.

        Truncates from the BEGINNING to keep the end of the window (closest
        to the prediction position).

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors of shape
            (len(windows), max_seq_length)
        """
        encoded = self.tokenizer(
            windows,
            padding=False,
            truncation=False,
            return_tensors=None,
            add_special_tokens=True,
        )

        input_ids_list = []
        attention_mask_list = []

        for ids in encoded["input_ids"]:
            if len(ids) > self.max_seq_length:
                ids = ids[-self.max_seq_length :]
            pad_len = self.max_seq_length - len(ids)
            input_ids_list.append([self.pad_token_id] * pad_len + ids)
            attention_mask_list.append([0] * pad_len + [1] * len(ids))

        # Create on CPU then move to GPU in one bulk transfer
        # (faster than torch.tensor(..., device="cuda") from Python lists)
        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long).to(
                self.device
            ),
        }

    def pad_to_batch_size(self, windows: list[str]) -> tuple[list[str], int]:
        """Pad window list to a multiple of batch_size."""
        n = len(windows)
        remainder = n % self.batch_size
        if remainder != 0:
            pad_count = self.batch_size - remainder
            return windows + [windows[0]] * pad_count, pad_count
        return windows, 0

    def embed(self, windows: list[str], normalize: bool = True) -> torch.Tensor:
        """
        Embed a list of windows.

        Returns:
            Tensor of shape (num_windows, 768) on device, dtype float16
        """
        n = len(windows)
        if n == 0:
            raise ValueError("No windows to embed")

        windows, pad_count = self.pad_to_batch_size(windows)
        features = self.tokenize(windows)

        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                batch_features = {
                    "input_ids": input_ids[i : i + self.batch_size],
                    "attention_mask": attention_mask[i : i + self.batch_size],
                }
                out = self.model.forward(batch_features)
                emb = out["sentence_embedding"]
                if normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb)

        embeddings = torch.cat(all_embeddings, dim=0)
        if pad_count > 0:
            embeddings = embeddings[:n]

        return embeddings.half()
