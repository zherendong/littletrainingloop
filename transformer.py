"""
Transformer model.
"""

import dataclasses
from torch.utils import checkpoint
import torch
import torch.nn as nn
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
    glu: bool = False
    nonlinearity: str = "gelu"
    embedding_norm: bool = True
    inner_size_multiple_of: int = 256  # Rounding for MLP hidden size (use 64 for GLU parameter parity)

    # experimental architectural choices
    pre_projection_transform: str | None = (
        # one of "proj_down", "down_add", "down_select", "down_add_128"
        "proj_down"
    )
    pre_projection_factor: float = 0.5
    early_mlp_scaling: float = 1
    middle_mlp_scaling: float = 1
    late_mlp_scaling: float = 1
    crown_mlp_scaling: float = 1  # https://arxiv.org/pdf/2509.06518#page=3.45
    early_attention_scaling: float = 1
    middle_attention_scaling: float = 1
    late_attention_scaling: float = 1
    crown_attention_scaling: float = 1
    segmented_norm: int | None = (
        None  # Try 128 and 512. The goal is that nothing changes.
    )
    skinny_mlps: bool = False
    skinny_queries: bool = False
    copy_values: bool = False
    # output scaling one of None, "scalar", "per_channel", "scalar0", "per_channel0", "per_channel_sigmoid", "per_channel_sigmoid_minus"
    output_scaling_mode: str | None = None

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

    def get_layer_stage(self, layer_idx: int) -> str:
        """Returns "early", "middle", or "late"."""
        if layer_idx == 0 or layer_idx == self.num_layers - 1:
            return "crown"
        is_early = layer_idx < self.num_layers / 3
        is_middle = layer_idx <= (self.num_layers * 2 / 3) and not is_early
        is_late = not (is_early or is_middle)
        if is_early:
            return "early"
        if is_middle:
            return "middle"
        if is_late:
            return "late"
        raise ValueError("unreachable")


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

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if self.segment_size:
            assert self.input_size % self.segment_size == 0
            num_segments = self.input_size // self.segment_size
            orig_shape = x.shape
            x = x.view(-1, num_segments, self.segment_size)
            x = self.norm(x)
            x *= 1.0 / num_segments
            x = x.view(*orig_shape)
        else:
            x = self.norm(x)
        return x.to(input_dtype)


def generate_fuzzy_diagonal(input_size: int, output_size: int, dtype):
    x = torch.arange(input_size, dtype=dtype).unsqueeze(1)
    y = torch.arange(output_size, dtype=dtype).unsqueeze(0)

    # compress to range from 0 to 1
    x = x / (input_size - 1)
    y = y / (output_size - 1)

    diff = 1 + 10 * torch.abs(x - y)
    diff_diag = 1 / (diff * diff)
    assert diff_diag.shape == (input_size, output_size)
    return diff_diag


class CopyLinear(nn.Module):
    """A linear layer encouraged to copy information."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype,
    ):
        super(CopyLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size, bias=False, dtype=dtype)

        with torch.no_grad():
            # initialization
            # init with variance 1/fan_in
            # self.linear.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
            # We add the diagonal instead of assigning it directly to get some sort of variance away from the diagonal
            pre_std = torch.std(self.linear.weight)
            print(f"{pre_std=}")
            self.linear.weight += generate_fuzzy_diagonal(
                output_size, input_size, dtype=dtype
            )
            # renormalize to std
            post_std = torch.std(self.linear.weight)
            print(f"{post_std=}")
            self.linear.weight *= pre_std / post_std
            re_std = torch.std(self.linear.weight)
            print(f"{re_std=}")

    def forward(self, x):
        x = self.linear(x)
        return x


class SkinnyLinear(nn.Module):
    """Down and up projection; never use bias.

    This module is a drop-in replacement of linear layers and meant to
    save half of the weights compared to a linear layer. So it defaults
    to using 1/4th of the input size as the inner size.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype,
        inner_size: int | None = None,
    ):
        super(SkinnyLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        assert self.input_size % 4 == 0
        self.compressed_size = inner_size or self.input_size // 4
        self.compressor = nn.Linear(
            input_size, self.compressed_size, bias=False, dtype=dtype
        )
        self.expander = nn.Linear(
            self.compressed_size, self.output_size, bias=False, dtype=dtype
        )

    def forward(self, x):
        x = self.compressor(x)
        x = self.expander(x)
        return x


class OutputScaling(nn.Module):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        mode: str | None = None,
    ):
        super(OutputScaling, self).__init__()
        self.size = size
        self.mode = mode

        if mode is None:
            pass
        elif mode == "scalar":
            self.scale = nn.Parameter(torch.ones(1, dtype=dtype))
        elif mode == "per_channel":
            self.scale = nn.Parameter(torch.ones(size, dtype=dtype))
        elif mode == "scalar0":
            self.scale = nn.Parameter(torch.zeros(1, dtype=dtype))
        elif mode == "per_channel0":
            self.scale = nn.Parameter(torch.zeros(size, dtype=dtype))
        elif mode == "per_channel_sigmoid":
            self.scale = nn.Parameter(torch.zeros(size, dtype=dtype))
        elif mode == "per_channel_sigmoid_minus":
            init_value = torch.zeros(size, dtype=dtype)
            init_value.fill_(-1.0)
            self.scale = nn.Parameter(init_value)
        else:
            raise ValueError(f"Unknown output scaling mode: {mode}")

    def forward(self, x):
        if self.mode is None:
            return x
        if self.mode in ["per_channel_sigmoid", "per_channel_sigmoid_minus"]:
            return x * torch.sigmoid(self.scale)
        return x * self.scale


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
        output_scaling_mode: str | None = None,
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

        if skinny:
            self.linear_in = SkinnyLinear(input_size, self.inner_size, dtype=dtype)
        else:
            self.linear_in = nn.Linear(
                input_size, self.inner_size, bias=False, dtype=dtype
            )

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
        else:
            raise NotImplementedError

        self.linear_out = nn.Linear(
            self.inner_size, self.output_size, bias=False, dtype=dtype
        )
        # with torch.no_grad():
        #     self.linear_out.weight.data.zero_()  # initialize as zero
        self.output_scaling = OutputScaling(
            self.output_size, dtype=dtype, mode=output_scaling_mode
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

        if self.glu:
            x1 = self.linear_in(x)
            x2 = self.linear_glu(x)
            x = self.nonlinearity(x1) * x2
        else:
            x = self.linear_in(x)
            x = self.nonlinearity(x)

        x = self.linear_out(x)
        x = self.output_scaling(x)
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
        segmented_norm: int | None = None,
        skinny_queries: bool = False,
        copy_values: bool = False,
        output_scaling_mode: str | None = None,
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
        if skinny_queries:
            self.linear_q = SkinnyLinear(
                input_size,
                num_heads_q * head_dim,
                dtype=dtype,
            )
        else:
            self.linear_q = nn.Linear(
                input_size,
                num_heads_q * head_dim,
                bias=False,
                dtype=dtype,
            )
        self.linear_k = nn.Linear(
            input_size,
            num_heads_kv * head_dim,
            bias=False,
            dtype=dtype,
        )
        if copy_values:
            self.linear_v = CopyLinear(
                input_size, num_heads_kv * self.head_dim_v, dtype=dtype
            )
        else:
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

        self.output_scaling = OutputScaling(
            self.input_size,
            dtype=dtype,
            mode=output_scaling_mode,
        )

        # with torch.no_grad():
        #     # initialization
        #     self.linear_q.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     self.linear_k.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     self.linear_v.weight.normal_(mean=0.0, std=1.0 / math.sqrt(input_size))
        #     self.linear_out.weight.normal_(
        #         mean=0.0, std=1.0 / math.sqrt(num_heads_q * head_dim_v)
        #     )

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
        out = self.output_scaling(out)
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

        layer_stage = config.get_layer_stage(block_idx)
        mlp_scaling_factor = {
            "early": config.early_mlp_scaling,
            "middle": config.middle_mlp_scaling,
            "late": config.late_mlp_scaling,
            "crown": config.crown_mlp_scaling,
        }[layer_stage]

        self.mlp = MLP(
            dtype=params_dtype,
            mlp_scaling_factor=mlp_scaling_factor,
            input_size=config.embedding_size,
            nonlinearity=config.nonlinearity,
            segmented_norm=config.segmented_norm,
            glu=config.glu,
            skinny=config.skinny_mlps,
            inner_size_multiple_of=config.inner_size_multiple_of,
            output_scaling_mode=config.output_scaling_mode,
        )

        attention_scaling_factor = {
            "early": config.early_attention_scaling,
            "middle": config.middle_attention_scaling,
            "late": config.late_attention_scaling,
            "crown": config.crown_attention_scaling,
        }[layer_stage]
        num_heads_q = int(config.num_heads * attention_scaling_factor)
        num_heads_kv = config.num_heads_kv
        if num_heads_q % num_heads_kv != 0:
            num_heads_kv = math.ceil(num_heads_q / 8)
            while num_heads_q % num_heads_kv != 0:
                if attention_scaling_factor > 1.0:
                    num_heads_kv += 1
                else:
                    num_heads_kv -= 1
        print(
            f"Block {block_idx} has {num_heads_q} heads_q and {num_heads_kv} heads_kv"
        )
        self.attention = SelfAttention(
            input_size=config.embedding_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=config.head_dim,
            use_flash_attention=config.use_flash_attention,
            dtype=params_dtype,
            segmented_norm=config.segmented_norm,
            skinny_queries=config.skinny_queries,
            copy_values=config.copy_values,
            output_scaling_mode=config.output_scaling_mode,
        )

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

        # with torch.no_grad():
        #     # initialization
        #     self.embedding.weight.normal_(mean=0.0, std=1.0 / math.sqrt(self.dim))
        #     self.output_projection.weight.normal_(
        #         mean=0.0, std=1.0 / math.sqrt(self.dim)
        #     )

        self._forward_opt = torch.compile(
            self._forward, mode="max-autotune", fullgraph=True
        )
        # self._forward_opt = self._forward

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
