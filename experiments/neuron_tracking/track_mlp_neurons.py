"""
Track dead / underused neurons in transformer MLPs during training.

Trains a standard transformer on character-level TinyShakespeare and, every
`log_every` steps, records per-neuron activation statistics for every MLP
layer.  A "neuron" here is one index in the MLP hidden dimension — i.e. one
key-value pair in the key-value-store view of an MLP.

Output CSV columns
------------------
step        : training step at which the stats were recorded
layer       : transformer layer index (0-based)
neuron      : neuron / KV-pair index within that layer's MLP hidden dim
mean_abs    : mean |activation| across all (batch × sequence) positions
pct_active  : fraction of (batch × sequence) positions where |activation| > threshold

The activation captured is the tensor that is fed into linear_out, i.e.
  - non-GLU:  act(linear_in(x))
  - GLU:      act(linear_in(x)) * linear_gate(x)

Usage
-----
python track_mlp_neurons.py                     # defaults
python track_mlp_neurons.py --n_steps 5000 --log_every 50 --glu
"""

import argparse
import csv
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Data ─────────────────────────────────────────────────────────────────────

def load_shakespeare(data_dir: str = "./data") -> str:
    path = Path(data_dir) / "shakespeare.txt"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        url = (
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt"
        )
        print(f"Downloading TinyShakespeare to {path}…")
        urllib.request.urlretrieve(url, path)
    return path.read_text()


def make_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


class ShakespeareDataset:
    def __init__(self, data: np.ndarray, seq_len: int, batch_size: int, seed: int = 42):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        max_start = len(self.data) - self.seq_len - 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Standard MLP (optionally GLU) that stores its pre-output activations."""

    def __init__(self, d_model: int, d_ff: int, glu: bool = False, activation: str = "relu"):
        super().__init__()
        self.glu = glu
        self.norm = nn.LayerNorm(d_model)
        self.linear_in = nn.Linear(d_model, d_ff, bias=False)
        if glu:
            self.linear_gate = nn.Linear(d_model, d_ff, bias=False)
        self.linear_out = nn.Linear(d_ff, d_model, bias=False)
        _acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "swish": nn.SiLU()}
        if activation not in _acts:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_acts)}")
        self.act = _acts[activation]
        # will hold the activation tensor after each forward pass
        self._last_activations: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        keys = self.linear_in(h)
        if self.glu:
            gates = self.linear_gate(h)
            acts = self.act(keys) * gates
        else:
            acts = self.act(keys)
        # Store for later stats collection (detached to avoid holding the graph)
        self._last_activations = acts.detach()
        return self.linear_out(acts)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(~self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        glu: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.mlp = MLP(d_model, d_ff, glu=glu, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        glu: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, max_seq_len, glu, activation)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = tok + pos
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)


# ── Activation stats ──────────────────────────────────────────────────────────

def collect_activation_stats(
    model: Transformer, threshold: float
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """
    Read the stored _last_activations from every MLP and compute per-neuron stats.

    Returns
    -------
    dict mapping layer_idx -> (mean_abs, pct_active)
      both tensors have shape (d_ff,) and are on CPU.
    """
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx, block in enumerate(model.blocks):
        acts = block.mlp._last_activations  # (B, T, d_ff)
        if acts is None:
            continue
        acts_f = acts.float().view(-1, acts.shape[-1])  # (B*T, d_ff)
        mean_abs = acts_f.abs().mean(dim=0).cpu()         # (d_ff,)
        pct_active = (acts_f.abs() > threshold).float().mean(dim=0).cpu()  # (d_ff,)
        stats[layer_idx] = (mean_abs, pct_active)
    return stats


def write_stats(
    writer: csv.writer,
    step: int,
    stats: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for layer_idx, (mean_abs, pct_active) in stats.items():
        for neuron_idx in range(len(mean_abs)):
            writer.writerow(
                [
                    step,
                    layer_idx,
                    neuron_idx,
                    f"{mean_abs[neuron_idx].item():.6f}",
                    f"{pct_active[neuron_idx].item():.4f}",
                ]
            )


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_loss(
    model: Transformer, dataset: ShakespeareDataset, vocab_size: int, n_batches: int, device: str
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.next_batch()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
    model.train()
    return total_loss / n_batches


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    n_steps: int = 2000,
    batch_size: int = 64,
    seq_len: int = 128,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 2048,
    glu: bool = False,
    activation: str = "relu",
    lr: float = 3e-4,
    log_every: int = 10,
    val_batches: int = 20,
    threshold: float = 0.1,
    output_csv: str = "./results/mlp_neuron_stats.csv",
    data_dir: str = "./data",
    device: str | None = None,
    seed: int = 42,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    text = load_shakespeare(data_dir)
    stoi, itos, vocab_size = make_vocab(text)
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    n_train = int(0.9 * len(data))
    dataset = ShakespeareDataset(data[:n_train], seq_len, batch_size, seed=seed)
    val_dataset = ShakespeareDataset(data[n_train:], seq_len, batch_size, seed=seed + 1)
    print(f"Vocab size: {vocab_size}  |  Train tokens: {n_train:,}  |  Val tokens: {len(data) - n_train:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        glu=glu,
        activation=activation,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"MLP hidden dim (d_ff): {d_ff}  |  Layers: {n_layers}")
    print(
        f"Rows per log step: {n_layers * d_ff:,}  |  "
        f"Total CSV rows (approx): {(n_steps // log_every) * n_layers * d_ff:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Loop ──────────────────────────────────────────────────────────────────
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "layer", "neuron", "mean_abs", "pct_active"])

        for step in range(n_steps):
            x, y = dataset.next_batch()
            x, y = x.to(device), y.to(device)

            logits = model(x)  # forward — also populates _last_activations
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_every == 0:
                stats = collect_activation_stats(model, threshold=threshold)
                write_stats(writer, step, stats)

            if step % (log_every * 20) == 0:
                val_loss = compute_val_loss(model, val_dataset, vocab_size, val_batches, device)
                print(f"Step {step:>6}/{n_steps}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

    print(f"\nDone. Activation stats saved to {out_path}")
    print(
        "Tip: load the CSV with pandas and groupby(['layer', 'neuron']) "
        "to see how each KV pair's usage evolves over training."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_steps",    type=int,   default=8000,   help="Training steps (default:8000)")
    p.add_argument("--batch_size", type=int,   default=64,     help="Batch size (default: 64)")
    p.add_argument("--seq_len",    type=int,   default=128,    help="Sequence length (default: 128)")
    p.add_argument("--d_model",    type=int,   default=128,    help="Model dimension (default: 128)")
    p.add_argument("--n_heads",    type=int,   default=4,      help="Attention heads (default: 4)")
    p.add_argument("--n_layers",   type=int,   default=2,      help="Transformer layers (default: 2)")
    p.add_argument("--d_ff",       type=int,   default=2048,    help="MLP hidden dim / # KV pairs (default: 2048)")
    p.add_argument("--glu",        action="store_true",        help="Use Gated Linear Unit in MLP")
    p.add_argument("--activation", type=str,   default="relu", help="Activation fn: relu|gelu|silu (default: relu)")
    p.add_argument("--lr",         type=float, default=3e-4,   help="Learning rate (default: 3e-4)")
    p.add_argument("--log_every",   type=int,   default=10,     help="Log stats every N steps (default: 10)")
    p.add_argument("--val_batches", type=int,   default=20,     help="Batches to use when computing val loss (default: 20)")
    p.add_argument("--threshold",  type=float, default=0.1,   help="|activation| threshold for pct_active (default: 0.1)")
    p.add_argument("--output_csv", type=str,   default="./results/mlp_neuron_stats.csv")
    p.add_argument("--data_dir",   type=str,   default="./data")
    p.add_argument("--device",     type=str,   default=None,   help="cuda|cpu (default: cuda)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
