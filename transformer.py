"""
Transformer model.
"""

import dataclasses
from typing import Callable
import math
from torch.utils import checkpoint
import torch
import torch.nn as nn
import attention
import torchtune

import language_model_basics


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    num_layers: int = 8
    num_heads: int = 8
    num_heads_kv: int = 0  # 0 = automatic
    head_dim: int = 64
    mlp_inner_size: int = 512
    embedding_size: int = 128
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.num_heads_kv == 0:
            # automatic GQA
            # num_heads_kv = math.ceil(self.num_heads / 8)
            # object.__setattr__(
            #     self,
            #     "num_heads_kv",
            #     num_heads_kv,
            # )
            # automatic MHA
            object.__setattr__(
                self,
                "num_heads_kv",
                self.num_heads,
            )


class ConfigRegistry:
    def __init__(self):
        self._configs = {}

    def register(self, name: str):
        """Decorator to register a config factory function."""
        if " " in name:
            raise ValueError(f"Config name {name} must not contain spaces")

        def decorator(func):
            config = func()
            self._configs[name] = config
            return config

        return decorator

    def get(self, name: str) -> TransformerConfig:
        return self._configs[name]

    def list_configs(self):
        return list(self._configs.keys())


transformer_config_registry = ConfigRegistry()


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

    # @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self.norm(x)
        return x.to(input_dtype)


# TODO: GLU variant
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        inner_size: int | None = None,
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

        # with torch.no_grad():
        #     # initialization
        #     # init with variance 1/fan_in
        #     self.linear_in.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     # TODO: use smaller variance for initializing the output layer,
        #     # a la Section D.2 in the mu parameterization paper
        #     self.linear_out.weight.normal_(mean=0.0, std=1.0 / math.sqrt(inner_size))

    def forward(self, x):
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.nonlinearity(x)
        x = self.linear_out(x)
        assert x.dtype == torch.bfloat16
        return x


class SelfAttention(nn.Module):  # non-flash
    def __init__(
        self,
        input_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        use_flash_attention: bool,
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

        self.use_flash_attention = use_flash_attention

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
        self.rotary_emb = torchtune.modules.RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=8192
        )

        # with torch.no_grad():
        #     # initialization
        #     self.linear_q.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     self.linear_k.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     self.linear_v.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     self.linear_out.weight.normal_(
        #         mean=0.0, std=1.0 / math.sqrt(num_heads_q * head_dim_v)
        #     )

    # @torch.compile(mode="max-autotune", fullgraph=True)
    def forward(self, x):
        x = self.norm(x)
        batch_size, sequence_length, _ = x.shape
        q = self.linear_q(x).view(
            batch_size, sequence_length, self.num_heads_q, self.head_dim
        )
        k = self.linear_k(x).view(
            batch_size, sequence_length, self.num_heads_kv, self.head_dim
        )
        v = self.linear_v(x).view(
            batch_size, sequence_length, self.num_heads_kv, self.head_dim_v
        )

        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        assert q.dtype == x.dtype

        q = q.view(
            batch_size, sequence_length, self.num_heads_kv, self.q_per_kv, self.head_dim
        )

        out = attention.attention_fn(q, k, v, use_flash=self.use_flash_attention)
        out = out.reshape(
            batch_size, sequence_length, self.num_heads_q * self.head_dim_v
        )
        out = self.linear_out(out)
        assert out.dtype == torch.bfloat16
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        block_idx: int,
        input_size: int,
        mlp_factory: Callable[[], nn.Module],
        attention_factory: Callable[[], nn.Module],
    ):
        super(TransformerBlock, self).__init__()
        self.block_idx = block_idx
        self.input_size = input_size
        self.mlp = mlp_factory()
        self.attention = attention_factory()

    def _forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

    def forward(self, x: torch.Tensor):
        if self.block_idx % 2 == 0:
            x = checkpoint.checkpoint(
                self._forward,
                x,
                use_reentrant=False,
            )  # type: ignore
            return x
        return self._forward(x)


class TransformerModel(language_model_basics.LanguageModel):
    """Simple transformer model."""

    def __init__(self, vocab_size: int, config: TransformerConfig):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.dim = config.embedding_size
        self.embedding = nn.Embedding(vocab_size, self.dim, dtype=torch.bfloat16)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    i,
                    self.dim,
                    lambda: MLP(self.dim, inner_size=config.mlp_inner_size),
                    lambda: SelfAttention(
                        self.dim,
                        config.num_heads,
                        config.num_heads_kv,
                        config.head_dim,
                        use_flash_attention=config.use_flash_attention,
                    ),
                )
                for i in range(config.num_layers)
            ]
        )

        # output projection
        self.final_norm = FP32LayerNorm(self.dim)
        self.output_projection = nn.Linear(
            self.dim,
            vocab_size,
            dtype=torch.bfloat16,
            bias=False,
        )
        print(
            f"Num non-embedding parameters: {self.num_non_embedding_parameters()} parameters"
        )

        # with torch.no_grad():
        #     # initialization
        #     self.embedding.weight.normal_(mean=0.0, std=1.0 / math.sqrt(self.dim))
        #     self.output_projection.weight.normal_(
        #         mean=0.0, std=1.0 / math.sqrt(self.dim)
        #     )

    def forward(self, x: torch.Tensor):
        """Returns embeddings, not logits.

        Use get_output_projection_weights to get the weights to compute the logits.
        """
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return x

    def get_output_projection_weights(self) -> torch.Tensor:
        return self.output_projection.weight

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_embedding_parameters(self):
        return sum(p.numel() for p in self.embedding.parameters())

    def num_non_embedding_parameters(self):
        return self.num_parameters() - self.num_embedding_parameters()
