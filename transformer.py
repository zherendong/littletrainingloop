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


class FP32LayerNorm(nn.Module):
    def __init__(self, input_size: int):
        super(FP32LayerNorm, self).__init__()
        self.input_size = input_size
        self.norm = nn.LayerNorm(
            input_size,
            eps=1e-5,  # supposedly helps with stability
            bias=False,
            dtype=torch.float32,
        )

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self.norm(x)
        return x.to(input_dtype)


class MLP(nn.Module):
    def __init__(
        self, input_size: int, output_size: int = None, inner_size: int = None
    ):
        if inner_size is None:
            inner_size = input_size * 4
        if output_size is None:
            output_size = input_size
        super(MLP, self).__init__()
        self.norm = FP32LayerNorm(input_size)
        self.linear_in = nn.Linear(
            input_size, inner_size, bias=False, dtype=torch.bfloat16
        )
        self.nonlinearity = nn.ReLU()
        self.linear_out = nn.Linear(
            inner_size, output_size, bias=False, dtype=torch.bfloat16
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.nonlinearity(x)
        x = self.linear_out(x)
        assert x.dtype == torch.bfloat16
        return x


def attention_fn(q, k, v, use_flash: bool = True):
    if use_flash:
        try:
            from flash_attn import flash_attn_func

            # # q: (batch_size, seqlen, nheads, headdim)
            batch_size, sequence_length_q, num_heads_kv, q_per_kv, head_dim = q.shape
            num_heads_q = num_heads_kv * q_per_kv
            q = q.view(batch_size, sequence_length_q, num_heads_q, head_dim)
            q_per_kv = num_heads_q // num_heads_kv
            sequence_length_kv = k.shape[1]
            k = k.view(batch_size, sequence_length_kv, num_heads_kv, head_dim)
            v = v.view(batch_size, sequence_length_kv, num_heads_kv, head_dim)
            return flash_attn_func(q, k, v, causal=True).view(
                batch_size, sequence_length_q, num_heads_kv, q_per_kv, head_dim
            )
        except ImportError:
            pass
    batch_size, sequence_length_q, num_heads_kv, q_per_kv, head_dim = q.shape
    num_heads_q = num_heads_kv * q_per_kv
    sequence_length_kv = k.shape[1]
    # Names for einsum: b, t, h, q, d
    q *= 1 / head_dim**0.5
    scores = torch.einsum("bthqd,bThd->btThq", q, k)
    assert (
        sequence_length_q == sequence_length_kv
    ), "Need to test the triangle code path if this doesn't hold."
    causal_mask = torch.triu(
        torch.ones(
            sequence_length_q, sequence_length_kv, device=q.device, dtype=q.dtype
        )
        * float("-inf"),
        diagonal=1,
    ).view(1, sequence_length_q, sequence_length_kv, 1, 1)
    scores += causal_mask
    probs = torch.softmax(scores, dim=2)
    out = torch.einsum("btThq,bThd->bthqd", probs, v)
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
        assert (
            self.q_per_kv * num_heads_kv == num_heads_q
        ), "num_heads_q must be a multiple of num_heads_kv"
        self.norm = FP32LayerNorm(input_size)
        self.linear_q = nn.Linear(
            input_size, num_heads_q * head_dim, bias=False, dtype=torch.bfloat16
        )
        self.linear_k = nn.Linear(
            input_size, num_heads_kv * head_dim, bias=False, dtype=torch.bfloat16
        )
        self.linear_v = nn.Linear(
            input_size, num_heads_kv * head_dim_v, bias=False, dtype=torch.bfloat16
        )
        self.linear_out = nn.Linear(
            num_heads_q * head_dim_v, input_size, bias=False, dtype=torch.bfloat16
        )
        # rotary_emb = RotaryEmbedding(head_dim)
        # self.register_buffer("rotary_emb", rotary_emb)

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
        assert out.dtype == torch.bfloat16
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
        assert x.dtype == torch.bfloat16
        return x


class TransformerModel(nn.Module):
    """Simple transformer model: y = Wx + b"""

    def __init__(self, vocab_size: int, config: TransformerConfig):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.dim = config.embedding_size
        self.embedding = nn.Embedding(vocab_size, self.dim, dtype=torch.bfloat16)
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
        self.output_projection = nn.Linear(
            self.dim,
            vocab_size,
            dtype=torch.bfloat16,
            bias=False,
        )
        # print number of parameters
        print(
            f"Num non-embedding parameters: {sum(p.numel() for p in self.transformer_blocks.parameters())} parameters"
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output_projection(x)
        assert x.dtype == torch.bfloat16
        return x
