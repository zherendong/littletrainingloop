"""
Transformer model.
"""

import dataclasses
from typing import Callable

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    num_layers: int = 8
    num_heads: int = 8
    num_heads_kv: int = 2
    head_dim: int = 64
    mlp_inner_size: int = 512
    embedding_size: int = 128


class MLP(nn.Module):
    def __init__(
        self, input_size: int, output_size: int = None, inner_size: int = None
    ):
        if inner_size is None:
            inner_size = input_size * 4
        if output_size is None:
            output_size = input_size
        super(MLP, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.linear_in = nn.Linear(input_size, inner_size)
        self.nonlinearity = nn.ReLU()
        self.linear_out = nn.Linear(inner_size, output_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.nonlinearity(x)
        x = self.linear_out(x)
        return x


def attention_fn(q, k, v):
    # Names for einsum: b, t, h, q, d
    batch_size, sequence_length, num_heads_kv, q_per_kv, head_dim = q.shape
    q *= 1 / head_dim**0.5
    scores = torch.einsum("bthqd,bThd->btThq", q, k)
    causal_mask = torch.triu(
        torch.ones(sequence_length, sequence_length) * float("-inf"), diagonal=1
    ).view(1, sequence_length, sequence_length, 1, 1)
    scores += causal_mask
    scores = torch.softmax(scores, dim=-4)
    out = torch.einsum("btThq,bThd->bthqd", scores, v)
    return out


class SelfAttention(nn.Module):  # non-flash
    def __init__(
        self,
        input_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        head_dim_v: int | None = None,
    ):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        if head_dim_v is None:
            head_dim_v = head_dim
        self.head_dim_v = head_dim_v
        self.q_per_kv = num_heads_q // num_heads_kv
        assert self.q_per_kv * num_heads_kv == num_heads_q, (
            "num_heads_q must be a multiple of num_heads_kv"
        )
        self.norm = nn.LayerNorm(input_size)
        self.linear_q = nn.Linear(input_size, num_heads_q * head_dim)
        self.linear_k = nn.Linear(input_size, num_heads_kv * head_dim)
        self.linear_v = nn.Linear(input_size, num_heads_kv * head_dim_v)
        self.linear_out = nn.Linear(num_heads_q * head_dim_v, input_size)

    def forward(self, x):
        x = self.norm(x)
        batch_size, sequence_length, _ = x.shape
        q = self.linear_q(x).view(
            batch_size, sequence_length, self.num_heads_kv, self.q_per_kv, self.head_dim
        )
        k = self.linear_k(x).view(
            batch_size, sequence_length, self.num_heads_kv, self.head_dim
        )
        v = self.linear_v(x).view(
            batch_size, sequence_length, self.num_heads_kv, self.head_dim_v
        )

        out = attention_fn(q, k, v)
        out = out.reshape(
            batch_size, sequence_length, self.num_heads_q * self.head_dim_v
        )
        out = self.linear_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_size,
        mlp_factory: Callable[[], nn.Module],
        attention_factory: Callable[[], nn.Module],
    ):
        super(TransformerBlock, self).__init__()
        self.input_size = input_size
        self.mlp = mlp_factory()
        self.attention = attention_factory()

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x


class TransformerModel(nn.Module):
    """Simple transformer model: y = Wx + b"""

    def __init__(self, vocab_size: int, seed: int, config: TransformerConfig):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.dim = config.embedding_size
        with torch.random.fork_rng():
            torch.random.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, self.dim)
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        self.dim,
                        lambda: MLP(self.dim, inner_size=config.mlp_inner_size),
                        lambda: SelfAttention(
                            self.dim,
                            config.num_heads,
                            config.num_heads_kv,
                            config.head_dim,
                        ),
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.output_projection = nn.Linear(self.dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_projection(x)
        return x
