"""
Transformer model.
"""

import dataclasses
from torch.utils import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune
import math

import attention
import cross_entropy
import language_model_basics

torch._dynamo.config.cache_size_limit = 64


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    num_layers: int = 8
    num_heads: int = 8
    num_heads_kv: int = 0  # 0 = automatic
    head_dim: int = 64
    mlp_inner_size: int = 512
    embedding_size: int = 128
    use_flash_attention: bool = True

    # classic architectural options
    gqa: bool = True
    glu: bool = True
    # one of "swish", "gelu", "relu", "PolyReLU", "PolyNorm"
    nonlinearity: str = "swish"
    embedding_norm: bool = True
    inner_size_multiple_of: int = (
        256  # Rounding for MLP hidden size (use 64 for GLU parameter parity)
    )

    # experimental architectural choices
    pre_projection_transform: str | None = (
        # one of "proj_down", "down_add", "down_select", "down_add_128"
        "proj_down"
    )
    pre_projection_factor: float = 0.5
    segmented_norm: int | None = (
        None  # Try 128 and 512. The goal is that nothing changes.
    )
    skinny_mlps: bool = False
    skinny_queries: bool = False
    tt_init: bool = False
    depth_init: bool = False
    final_proj_init_std: float = 1.0

    def __post_init__(self):
        if self.num_heads_kv == 0:
            num_heads_kv = self.num_heads
            if self.gqa:
                num_heads_kv = math.ceil(self.num_heads / 8)
                while self.num_heads % num_heads_kv != 0:
                    num_heads_kv += 1
                assert self.num_heads >= num_heads_kv
            object.__setattr__(
                self,
                "num_heads_kv",
                num_heads_kv,
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
    def __init__(self, input_size: int, segment_size: int | None = None):
        super(FP32LayerNorm, self).__init__()
        self.input_size = input_size
        self.segment_size = segment_size
        self.norm = nn.LayerNorm(
            segment_size or input_size,
            eps=1e-5,  # supposedly helps with stability
            bias=False,
            dtype=torch.float32,
        )

    def init_weights(self):
        self.norm.reset_parameters()

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if self.segment_size:
            assert self.input_size % self.segment_size == 0
            num_segments = self.input_size // self.segment_size
            orig_shape = x.shape
            x = x.view(-1, num_segments, self.segment_size)
            x = self.norm(x)
            x = x.view(*orig_shape)
        else:
            x = self.norm(x)
        return x.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def init_weights(self):
        nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FlexLinear(nn.Module):
    """Can implement Linear or down and up projection.

    This module is a drop-in replacement of linear layers and meant to
    save half of the weights compared to a linear layer. So it defaults
    to using 1/4th of the input size as the inner size.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype,
        bias: bool = False,
        inner_size: int | None = None,
        is_skinny: bool = False,
    ):
        super(FlexLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.is_skinny = is_skinny

        if is_skinny:
            assert self.input_size % 4 == 0
            self.compressed_size = inner_size or self.input_size // 4
            self.compressor = nn.Linear(
                input_size, self.compressed_size, bias=self.bias, dtype=dtype
            )
            self.expander = nn.Linear(
                self.compressed_size, self.output_size, bias=self.bias, dtype=dtype
            )
        else:
            self.linear = nn.Linear(
                input_size, self.output_size, bias=self.bias, dtype=dtype
            )

    def init_weights(self, init_std: float):
        if not self.is_skinny:
            nn.init.trunc_normal_(self.linear.weight, mean=0.0, std=init_std)
        else:
            nn.init.trunc_normal_(self.compressor.weight, mean=0.0, std=init_std)
            nn.init.trunc_normal_(
                self.expander.weight, mean=0.0, std=init_std
            )  # TODO: divide init_std by compression factor?

    def forward(self, x):
        if not self.is_skinny:
            return self.linear(x)
        x = self.compressor(x)
        x = self.expander(x)
        return x


def _poly(x, weight, bias, order=3):
    return sum(weight[i] * (x ** (i + 1)) for i in range(order)) + bias


class PolyReLU(nn.Module):
    """Polynomial ReLU.

    https://arxiv.org/pdf/2411.03884
    """

    def __init__(self):
        super(PolyReLU, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(3) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, checkpointing=False):
        x = F.relu(x)
        if checkpointing:
            return checkpoint.checkpoint(
                _poly, x, self.weight, self.bias, use_reentrant=False
            )
        return _poly(x, self.weight, self.bias)


def _norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _poly_norm(x, weight, bias, order=3):
    return sum(weight[i] * _norm(x ** (i + 1)) for i in range(order)) + bias


class PolyNorm(nn.Module):
    """Polynomial normalization.

    https://arxiv.org/pdf/2411.03884
    """

    def __init__(self):
        super(PolyNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(3) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, checkpointing=False):
        if checkpointing:
            return checkpoint.checkpoint(
                _poly_norm, x, self.weight, self.bias, use_reentrant=False
            )
        return _poly_norm(x, self.weight, self.bias)


class MLP(nn.Module):
    """MLP with optional GLU."""

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        input_size: int,
        mlp_scaling_factor: float | None = None,
        output_size: int | None = None,
        nonlinearity: str = "relu",
        segmented_norm: int | None = None,
        glu: bool = False,
        skinny: bool = False,
        inner_size_multiple_of: int = 256,
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.inner_size = input_size * 4
        if mlp_scaling_factor:
            self.inner_size = int(self.inner_size * mlp_scaling_factor)
        if glu:
            self.inner_size = int(self.inner_size * 2 / 3)
        self.inner_size = inner_size_multiple_of * (
            (self.inner_size + inner_size_multiple_of - 1) // inner_size_multiple_of
        )
        print(
            f"MLP inner size: {self.inner_size}, {mlp_scaling_factor=}, {glu=}, {inner_size_multiple_of=}"
        )
        self.output_size = output_size or input_size
        del output_size
        self.segmented_norm = segmented_norm
        self.glu = glu
        self.skinny = skinny

        self.norm = FP32LayerNorm(input_size, segment_size=segmented_norm)

        self.linear_in = nn.Linear(input_size, self.inner_size, bias=False, dtype=dtype)

        if glu:
            self.linear_glu = nn.Linear(
                input_size, self.inner_size, bias=False, dtype=dtype
            )
        if nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "gelu":
            self.nonlinearity = nn.GELU()
        elif nonlinearity == "swish":
            self.nonlinearity = nn.SiLU()
        elif nonlinearity == "PolyNorm":
            self.nonlinearity = PolyNorm()
        elif nonlinearity == "PolyReLU":
            self.nonlinearity = PolyReLU()
        else:
            raise NotImplementedError

        self.linear_out = nn.Linear(
            self.inner_size, self.output_size, bias=False, dtype=dtype
        )

    def init_weights(self, init_std: float):
        self.norm.init_weights()
        nn.init.trunc_normal_(self.linear_in.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_out.weight, mean=0.0, std=init_std)
        if self.glu:
            nn.init.trunc_normal_(self.linear_glu.weight, mean=0.0, std=init_std)

    def forward(self, x):
        if self.glu:
            x = self.norm(x)
            x1 = self.linear_in(x)
            x2 = self.linear_glu(x)
            x = self.nonlinearity(x1) * x2
            x = self.linear_out(x)
            return x
        else:
            x = self.norm(x)
            x = self.linear_in(x)
            x = self.nonlinearity(x)
            x = self.linear_out(x)
            return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        use_flash_attention: bool,
        dtype: torch.dtype,
        head_dim_v: int | None = None,
        skinny_queries: bool = False,
    ):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.q_per_kv = num_heads_q // num_heads_kv

        assert (
            self.q_per_kv * num_heads_kv == num_heads_q
        ), f"num_heads_q must be a multiple of num_heads_kv, but got {num_heads_q=} and {num_heads_kv=} and {self.q_per_kv=}"

        self.use_flash_attention = use_flash_attention

        self.norm = FP32LayerNorm(self.input_size)
        self.linear_q = FlexLinear(
            input_size,
            num_heads_q * head_dim,
            bias=False,
            dtype=dtype,
            is_skinny=skinny_queries,
        )
        self.linear_k = nn.Linear(
            input_size,
            num_heads_kv * head_dim,
            bias=False,
            dtype=dtype,
        )
        self.linear_v = nn.Linear(
            input_size,
            num_heads_kv * self.head_dim_v,
            bias=False,
            dtype=dtype,
        )
        self.linear_out = nn.Linear(
            num_heads_q * self.head_dim_v,
            input_size,
            bias=False,
            dtype=dtype,
        )

        self.rotary_emb = torchtune.modules.RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=8192
        )

    def init_weights(self, init_std: float):

        self.norm.init_weights()

        # fixed std for q, k, and v like in Qwen
        # https://github.com/pytorch/torchtitan/blob/bb308da6bd85ed31a2670993326322e3631af436/torchtitan/models/qwen3/model/model.py#L177C13-L177C69
        self.linear_q.init_weights(0.02)
        nn.init.trunc_normal_(self.linear_k.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_v.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_out.weight, mean=0.0, std=init_std)

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
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        block_idx: int,
        params_dtype,
    ):
        super(TransformerBlock, self).__init__()
        self.block_idx = block_idx
        self.config = config

        self.mlp = MLP(
            dtype=params_dtype,
            input_size=config.embedding_size,
            nonlinearity=config.nonlinearity,
            segmented_norm=config.segmented_norm,
            glu=config.glu,
            skinny=config.skinny_mlps,
            inner_size_multiple_of=config.inner_size_multiple_of,
        )

        print(
            f"Block {block_idx} has {config.num_heads} heads_q and {config.num_heads_kv} heads_kv"
        )
        self.attention = SelfAttention(
            input_size=config.embedding_size,
            num_heads_q=config.num_heads,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            use_flash_attention=config.use_flash_attention,
            dtype=params_dtype,
            skinny_queries=config.skinny_queries,
        )

    def init_weights(self):
        print(f"Initializing block {self.block_idx}")
        # Initialization scheme from torchtitan/Qwen
        # compare https://github.com/pytorch/torchtitan/blob/bb308da6bd85ed31a2670993326322e3631af436/torchtitan/models/qwen3/model/model.py#L326
        if self.config.depth_init:
            init_std = 0.02 / (2 * (self.block_idx + 1)) ** 0.5
        else:
            init_std = 0.02 / (2 * self.config.num_layers) ** 0.5
        self.attention.init_weights(init_std)
        self.mlp.init_weights(init_std)

    def _forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

    def forward(self, x: torch.Tensor):
        x = checkpoint.checkpoint(
            self._forward,
            x,
            use_reentrant=False,
            preserve_rng_state=False,
        )  # type: ignore
        return x


class TransformerModel(language_model_basics.LanguageModel):
    """Simple transformer model."""

    def __init__(
        self,
        vocab_size: int,
        config: TransformerConfig,
        params_dtype=torch.bfloat16,
        activation_dtype=torch.bfloat16,
    ):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.dim = config.embedding_size
        self.config = config
        self.params_dtype = params_dtype
        self.activation_dtype = activation_dtype
        self.embedding = nn.Embedding(
            vocab_size,
            self.dim,
            # Always float32 for embedding, as recommended
            # for optimal quality.
            dtype=torch.float32,
        )
        if config.embedding_norm:
            self.embedding_norm = FP32LayerNorm(self.dim)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(config, i, params_dtype)
                for i in range(config.num_layers)
            ]
        )

        # output projection
        proj_input_dim = self.dim
        layernorm_dim = self.dim
        if self.config.pre_projection_transform == "down_add":
            proj_input_dim = proj_input_dim // 2
            layernorm_dim = proj_input_dim
        if self.config.pre_projection_transform == "down_select":
            proj_input_dim = proj_input_dim // 2
            layernorm_dim = proj_input_dim
        if self.config.pre_projection_transform == "down_add_128":
            proj_input_dim = 128
            layernorm_dim = proj_input_dim
        if self.config.pre_projection_transform == "proj_down":
            proj_input_dim = int(self.dim * config.pre_projection_factor)
            self.output_compressor = nn.Linear(
                self.dim,
                proj_input_dim,
                dtype=self.params_dtype,
                bias=False,
            )
            layernorm_dim = self.dim

        self.final_norm = FP32LayerNorm(layernorm_dim)
        self.output_projection = nn.Linear(
            proj_input_dim,
            vocab_size,
            dtype=self.params_dtype,
            bias=False,
        )
        print(
            f"Num non-embedding parameters: {self.num_non_embedding_parameters()} parameters"
        )

        self._forward_opt = torch.compile(
            self._forward, mode="max-autotune", fullgraph=True
        )
        # self._forward_opt = self._forward

        if config.tt_init:
            self.init_weights()

    def init_weights(self):
        """Explicit weight initialization.

        Avoids double initialization when modifying weights of
        submodules via pytorch builtin `reset_parameters`.
        """
        print("Initializing weights - model")
        nn.init.normal_(self.embedding.weight)
        if self.config.embedding_norm:
            self.embedding_norm.init_weights()

        for block in self.transformer_blocks:
            block.init_weights()  # type: ignore

        self.final_norm.init_weights()

        cutoff_factor = 3
        out_projs = [self.output_projection]
        if hasattr(self, "output_compressor"):
            out_projs.append(self.output_compressor)
        for out_proj in out_projs:
            input_dim = out_proj.weight.shape[1]
            out_proj_std = self.config.final_proj_init_std * input_dim**-0.5
            nn.init.trunc_normal_(
                out_proj.weight,
                mean=0.0,
                std=out_proj_std,
                a=-cutoff_factor * out_proj_std,
                b=cutoff_factor * out_proj_std,
            )

    def emb_transformation(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the embeddings before the output projection - default is norm."""
        batch, seq_len, emb_dim = x.shape
        if self.config.pre_projection_transform == "proj_down":
            x = self.final_norm(x)
            x = self.output_compressor(x)
            return x  # stand-out return statement to enable norm before compression

        if self.config.pre_projection_transform is None:
            x = x
        elif self.config.pre_projection_transform == "down_add":
            x = x[..., : emb_dim // 2] + x[..., emb_dim // 2 :]
        elif self.config.pre_projection_transform == "down_select":
            x = x[..., : emb_dim // 2]
        elif self.config.pre_projection_transform == "down_add_128":
            x = x.view(batch, seq_len, 128, emb_dim // 128)
            x = x.sum(-1)
        else:
            raise NotImplementedError
        return self.final_norm(x)

    def _forward(self, x: torch.Tensor):
        """Returns embeddings, not logits.

        Use get_output_projection_weights to get the weights to compute the logits.
        """
        x = self.embedding(x).to(self.activation_dtype)
        x = self.transformer_blocks(x)
        x = self.emb_transformation(x)
        # don't apply the output projection, as it's handled differently in
        # compute_loss and forward.
        return x

    def get_output_projection_weights(self):
        return self.output_projection.weight

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert targets.dtype == torch.long
        final_emb = self._forward_opt(inputs)

        emb_dim = final_emb.shape[-1]
        # flatten batch and sequence length for cross entropy
        final_emb = final_emb.view(-1, emb_dim)
        targets = targets.view(-1)

        weights = self.get_output_projection_weights()

        # loss = cut_cross_entropy.linear_cross_entropy(
        #     e=final_emb,
        #     c=weights,
        #     targets=targets,
        #     ignore_index=cross_entropy.cross_entropy_ignore_index,
        #     filter_eps=torch.finfo(torch.float32).eps,
        #     # accum_e_fp32=True,
        #     # accum_c_fp32=True,
        #     # impl="cce_kahan_full_c",
        #     impl="torch_compile",
        # )

        loss = cross_entropy.cross_entropy_with_logits_by_segment(
            final_emb.clone(), weights, targets
        )
        return loss

    def forward(self, x: torch.Tensor):
        final_emb = self._forward_opt(x)
        logits = self.output_projection(final_emb)
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_embedding_parameters(self):
        return sum(p.numel() for p in self.embedding.parameters())

    def num_non_embedding_parameters(self):
        return self.num_parameters() - self.num_embedding_parameters()
