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
import initialization
import spelling_bee_embeddings
import language_model_dataloader
import fp32norm
from experiments.chunk_embeddings import cluster_assigner

# The cache size limit is very limited by default.
# This can crash or stall training runs.
torch._dynamo.config.cache_size_limit = 128


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    num_layers: int = 8
    num_heads: int = 8
    num_heads_kv: int = 0  # 0 = automatic
    head_dim: int = 64
    mlp_inner_size: int = 512
    embedding_size: int = 128
    use_flash_attention: bool = True
    max_seq_len: int = 8192

    # classic architectural options
    gqa: bool = True
    glu: bool = True
    # one of "swish", "gelu", "relu", "PolyReLU", "PolyNorm"
    nonlinearity: str = "swish"
    embedding_norm: bool = False
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
    skinny_queries: bool = False
    skinny_scaled_init: bool = False

    # spelling bee options
    spelling_bee: bool = False
    separate_token_embedding: bool = True
    char_embedding_norm: bool = True
    char_init_scale: float = 1.0
    spelling_bee_in_out_scale: float = 1.0
    spelling_bee_out: bool = False
    spelling_bee_out_scale: float = 1.0
    apply_rotary: bool = True
    char_embedding_norm_out: bool = False
    char_init_scale_out: float = 1.0
    apply_rotary_out: bool = False
    spelling_bee_max_characters: int = 16
    spelling_type: str = "full"  # one of "full", "dummy", "shuffled"
    spelling_bee_rotary_base: int = 10000

    # initialization options
    zheren_init: bool = True
    depth_init: bool = False
    pairwise_cancelling_init: bool = False

    # chunk embeddings (sparse memory via learned clustering)
    chunk_embeddings: bool = False
    chunk_num_clusters: int = 1_000_000  # Number of clusters (default 1M)
    chunk_window_size: int = 32  # Context window size for clustering
    chunk_stride: int = (
        8  # Stride for cluster assignment (1 = every token, 8 = every 8th)
    )
    chunk_init_scale: float = 0.0  # Initial scale for value embeddings (0 = zero-init)
    chunk_centroid_path: str | None = (
        None  # Path to centroid file (required if chunk_embeddings=True)
    )
    chunk_cache_dir: str | None = None  # Directory for disk cache of cluster IDs

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


class FlexLinear(nn.Module):
    """Can implement Linear or skinny projection.

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
        scaled_init: bool = False,
    ):
        super(FlexLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.is_skinny = is_skinny
        self.scaled_init = scaled_init

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

    def init_weights(self):
        if not self.is_skinny:
            initialization.init_linear(self.linear, activation="linear")
        else:
            scaling_factor = 1 / math.sqrt(2) if self.scaled_init else 1.0
            initialization.init_linear(
                self.compressor, activation="linear", scaling_factor=scaling_factor
            )
            initialization.init_linear(self.expander, activation="linear")

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
        self.weight = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, x, checkpointing=False):
        x = F.relu(x)
        if checkpointing:
            return checkpoint.checkpoint(
                _poly, x, self.weight, self.bias, use_reentrant=False
            )
        return _poly(x, self.weight, self.bias)


def _norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, order=3
):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    assert x.ndim == 3
    x = sum(weight[i] * _norm(x ** (i + 1)) for i in range(order))  # type: ignore
    if bias is not None:
        x += bias
    return x.to(orig_dtype)


class PolyNorm(nn.Module):
    """Polynomial normalization.

    https://arxiv.org/pdf/2411.03884
    """

    def __init__(self):
        super(PolyNorm, self).__init__()
        self.weight = torch.nn.Parameter(
            (torch.ones(3, dtype=torch.float32) / 3) * 1.41
        )  # 1.41 is experimentally determined to preserve the output variance.
        # self.weight = torch.nn.Parameter(torch.ones(3, dtype=torch.float32) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x, checkpointing=False):
        if checkpointing:
            return checkpoint.checkpoint(
                _poly_norm, x, self.weight, self.bias, use_reentrant=False
            )
        return _poly_norm(x, self.weight, self.bias)


class Segmented(nn.Module):
    def __init__(self, dim: int, segment_size: int = 128):
        super(Segmented, self).__init__()
        self.segment_size = segment_size
        self.num_segments = dim // segment_size
        assert self.num_segments * segment_size == dim
        self.dim = dim
        # self.layernorms = [
        #     FP32LayerNorm(segment_size) for _ in range(self.num_segments)
        # ]
        self.layernorms = [PolyNorm() for _ in range(self.num_segments)]
        self.layernorms = nn.ModuleList(self.layernorms)

    def forward(self, x):
        assert x.ndim == 3
        assert x.shape[1] % self.segment_size == 0
        orig_shape = x.shape
        x = x.view(x.shape[0], -1, self.num_segments, self.segment_size)

        for i in range(self.num_segments):
            x[:, :, i, :] = self.layernorms[i](x[:, :, i, :])

        return x.view(*orig_shape)


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
        inner_size_multiple_of: int = 256,
        pairwise_cancelling_init: bool = False,
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
        self.output_size = output_size or input_size
        del output_size
        self.segmented_norm = segmented_norm
        self.glu = glu
        self.pairwise_cancelling_init = pairwise_cancelling_init

        self.norm = fp32norm.FP32LayerNorm(input_size, segment_size=segmented_norm)
        self.linear_in = nn.Linear(input_size, self.inner_size, bias=False, dtype=dtype)

        if glu:
            self.linear_glu = nn.Linear(
                input_size, self.inner_size, bias=False, dtype=dtype
            )
        self.nonlinearity = nonlinearity
        if nonlinearity == "relu":
            self.nonlinear_fn = nn.ReLU()
        elif nonlinearity == "gelu":
            self.nonlinear_fn = nn.GELU()
        elif nonlinearity == "swish":
            self.nonlinear_fn = nn.SiLU()
        elif nonlinearity == "polynorm":
            self.nonlinear_fn = PolyNorm()
        elif nonlinearity == "polyrelu":
            self.nonlinear_fn = PolyReLU()
        elif nonlinearity == "segmented":
            self.nonlinear_fn = Segmented(self.inner_size)
        else:
            raise NotImplementedError(f"Unknown nonlinearity: {nonlinearity}")

        self.linear_out = nn.Linear(
            self.inner_size, self.output_size, bias=False, dtype=dtype
        )

    def init_weights(self):
        """Initialize weights with activation-aware and optionally depth-scaled initialization."""
        initialization.init_linear(
            self.linear_in,
            activation=self.nonlinearity,
            pairwise_mode="equal" if self.pairwise_cancelling_init else None,
        )
        if self.glu:
            initialization.init_linear(
                self.linear_glu,
                activation="linear",
                pairwise_mode="equal" if self.pairwise_cancelling_init else None,
            )

        initialization.init_linear(
            self.linear_out,
            pairwise_mode="opposing" if self.pairwise_cancelling_init else None,
        )

    def forward(self, x):
        if self.glu:
            x = self.norm(x)
            x1 = self.linear_in(x)
            x2 = self.linear_glu(x)
            x = self.nonlinear_fn(x1) * x2
            x = self.linear_out(x)
            return x
        else:
            x = self.norm(x)
            x = self.linear_in(x)
            x = self.nonlinear_fn(x)
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
        max_seq_len: int,
        head_dim_v: int | None = None,
        skinny_queries: bool = False,
        skinny_scaled_init: bool = False,
    ):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.q_per_kv = num_heads_q // num_heads_kv

        self.max_seq_len = max_seq_len

        assert (
            self.q_per_kv * num_heads_kv == num_heads_q
        ), f"num_heads_q must be a multiple of num_heads_kv, but got {num_heads_q=} and {num_heads_kv=} and {self.q_per_kv=}"

        self.use_flash_attention = use_flash_attention

        self.norm = fp32norm.FP32LayerNorm(self.input_size)
        self.linear_q = FlexLinear(
            input_size,
            num_heads_q * head_dim,
            bias=False,
            dtype=dtype,
            is_skinny=skinny_queries,
            scaled_init=skinny_scaled_init,
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
            dim=head_dim, max_seq_len=self.max_seq_len
        )

    def init_weights(self):
        """Initialize weights with proper variance-preserving initialization."""
        self.linear_q.init_weights()
        initialization.init_linear(self.linear_k, activation="linear")
        initialization.init_linear(self.linear_v, activation="linear")
        initialization.init_linear(self.linear_out)

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
            inner_size_multiple_of=config.inner_size_multiple_of,
            pairwise_cancelling_init=config.pairwise_cancelling_init,
        )

        # print(
        #     f"Block {block_idx} has {config.num_heads} heads_q and {config.num_heads_kv} heads_kv"
        # )
        self.attention = SelfAttention(
            input_size=config.embedding_size,
            num_heads_q=config.num_heads,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            use_flash_attention=config.use_flash_attention,
            dtype=params_dtype,
            max_seq_len=config.max_seq_len,
            skinny_queries=config.skinny_queries,
            skinny_scaled_init=config.skinny_scaled_init,
        )

    def init_weights(self):
        # print(f"Initializing block {self.block_idx}")
        self.attention.init_weights()
        self.mlp.init_weights()

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

        emb_dtype = torch.float32
        if config.spelling_bee:
            vocab = language_model_dataloader.get_default_tokenizer_vocab()
            self.embedding = spelling_bee_embeddings.SpellingBeeEmbedding(
                num_tokens=vocab_size,
                embedding_dim=self.dim,
                vocab=vocab,
                max_characters=config.spelling_bee_max_characters,
                weight_dtype=emb_dtype,
                separate_token_embedding=config.separate_token_embedding,
                character_norm=config.char_embedding_norm,
                char_init_scale=config.char_init_scale,
                apply_rotary=config.apply_rotary,
                scale=config.spelling_bee_in_out_scale,
                spelling_type=config.spelling_type,
                rotary_base=config.spelling_bee_rotary_base,
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                self.dim,
                # Always float32 for embedding, as recommended
                # for optimal quality.
                dtype=emb_dtype,
            )
        if config.embedding_norm:
            self.embedding_norm = fp32norm.FP32LayerNorm(self.dim)

        # Chunk embeddings (sparse memory via learned clustering)
        if config.chunk_embeddings:
            self.chunk_value_embedding = nn.Embedding(
                config.chunk_num_clusters,
                self.dim,
                dtype=emb_dtype,
            )
            # Learned gate for chunk embeddings (initialized to produce ~0 output)
            # This is a per-dimension learnable parameter that gets sigmoid-activated
            self.chunk_gate = nn.Parameter(torch.zeros(self.dim, dtype=emb_dtype))

            # Load cluster assigner for computing cluster IDs
            assert (
                config.chunk_centroid_path is not None
            ), "chunk_centroid_path must be set when chunk_embeddings=True"

            self.cluster_assigner = cluster_assigner.ClusterAssigner(
                config.chunk_centroid_path,
                window_size=config.chunk_window_size,
            )
            print(
                f"Chunk embeddings enabled: {config.chunk_num_clusters:,} clusters, "
                f"{self.chunk_value_embedding.weight.numel():,} params (with learned gate)"
            )

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

        self.final_norm = fp32norm.FP32LayerNorm(layernorm_dim)
        if config.spelling_bee_out:
            vocab = language_model_dataloader.get_default_tokenizer_vocab()
            self.embedding_out = spelling_bee_embeddings.SpellingBeeEmbedding(
                num_tokens=vocab_size,
                embedding_dim=proj_input_dim,
                vocab=vocab,
                max_characters=config.spelling_bee_max_characters,
                weight_dtype=self.params_dtype,
                separate_token_embedding=False,
                character_norm=config.char_embedding_norm_out,
                char_init_scale=config.char_init_scale_out,
                apply_rotary=config.apply_rotary_out,
                scale=config.spelling_bee_out_scale,
                spelling_type=config.spelling_type,
                rotary_base=config.spelling_bee_rotary_base,
            )
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

        if config.zheren_init:
            self.init_weights()

    def init_weights(self):
        """Explicit weight initialization.

        Avoids double initialization when modifying weights of
        submodules via pytorch builtin `reset_parameters`.
        """
        print("Initializing weights with Zheren's scheme")
        if self.config.spelling_bee:
            assert isinstance(
                self.embedding, spelling_bee_embeddings.SpellingBeeEmbedding
            )
            self.embedding.init_weights()
        else:
            assert isinstance(self.embedding, nn.Embedding)
            initialization.init_embedding(self.embedding)
        if self.config.embedding_norm:
            self.embedding_norm.init_weights(init_val=1 / math.sqrt(self.dim))

        if hasattr(self, "output_compressor"):
            initialization.init_linear(self.output_compressor, activation="linear")

        # Initialize chunk value embeddings (zero-init by default for residual addition)
        # Initialize gate to -5.0 so sigmoid(-5.0) ≈ 0.007 (gate starts nearly closed)
        if self.config.chunk_embeddings:
            with torch.no_grad():
                self.chunk_value_embedding.weight.zero_()
                if self.config.chunk_init_scale > 0:
                    self.chunk_value_embedding.weight.normal_(
                        mean=0.0, std=self.config.chunk_init_scale
                    )
                # Initialize gate to produce near-zero output initially
                self.chunk_gate.fill_(-5.0)

        for block in self.transformer_blocks:
            block.init_weights()  # type: ignore

        self.final_norm.init_weights()
        initialization.init_linear(self.output_projection)
        if self.config.spelling_bee_out:
            assert isinstance(
                self.embedding_out, spelling_bee_embeddings.SpellingBeeEmbedding
            )
            self.embedding_out.init_weights()

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

    def _get_cluster_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Compute or load cached cluster IDs for input token IDs.

        NOTE: This method is NOT torch.compile compatible. Call it before _forward_opt.

        Delegates to ClusterAssigner.get_cluster_ids() which handles striding,
        expansion, and disk caching.

        Args:
            x: Token IDs of shape (batch, seq_len)

        Returns:
            cluster_ids: Tensor of shape (batch, seq_len)
        """
        return self.cluster_assigner.get_cluster_ids(
            x,
            stride=self.config.chunk_stride,
            cache_dir=self.config.chunk_cache_dir,
        )

    def _forward(self, x: torch.Tensor, cluster_ids: torch.Tensor | None = None):
        """Returns embeddings, not logits.

        Use get_output_projection_weights to get the weights to compute the logits.

        Args:
            x: Token IDs of shape (batch, seq_len)
            cluster_ids: Cluster IDs of shape (batch, seq_len) for chunk embeddings.
                         Must be provided if chunk_embeddings is enabled.
        """
        x = self.embedding(x).to(self.activation_dtype)
        # Add gated chunk value embeddings if available
        if cluster_ids is not None:
            chunk_emb = self.chunk_value_embedding(cluster_ids).to(
                self.activation_dtype
            )
            # Apply learned gate (sigmoid activation, per-dimension)
            gate = torch.sigmoid(self.chunk_gate.to(self.activation_dtype))
            x = x + gate * chunk_emb
        if self.config.embedding_norm:
            x = self.embedding_norm(x)
        x = self.transformer_blocks(x)
        x = self.emb_transformation(x)
        # don't apply the output projection, as it's handled differently in
        # compute_loss and forward.
        return x

    def get_output_projection_weights(self):
        weights = self.output_projection.weight
        if self.config.spelling_bee_out:
            assert isinstance(
                self.embedding_out, spelling_bee_embeddings.SpellingBeeEmbedding
            )
            all_tokens = torch.arange(
                self.vocab_size, dtype=torch.int32, device=weights.device
            ).unsqueeze(
                0
            )  # add dummy batch dim
            char_embeddings = self.embedding_out.get_character_embeddings(all_tokens)
            char_embeddings = char_embeddings.squeeze(0)  # remove dummy batch dim
            weights = 0.5 * weights + 0.5 * char_embeddings
        return weights

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        assert targets.dtype == torch.long
        # Compute cluster IDs outside compiled region if chunk embeddings enabled
        cluster_ids = None
        if self.config.chunk_embeddings:
            cluster_ids = self._get_cluster_ids(inputs)
        final_emb = self._forward_opt(inputs, cluster_ids)

        emb_dim = final_emb.shape[-1]
        # flatten batch and sequence length for cross entropy
        final_emb = final_emb.view(-1, emb_dim)
        targets = targets.view(-1)

        weights = self.get_output_projection_weights()

        loss = cross_entropy.cross_entropy_with_logits_by_segment(
            final_emb.clone(), weights, targets
        )
        return loss

    def forward(
        self,
        x: torch.Tensor,
        use_optimized: bool = True,
    ):
        # Compute cluster IDs outside compiled region if chunk embeddings enabled
        cluster_ids = None
        if self.config.chunk_embeddings:
            cluster_ids = self._get_cluster_ids(x)
        if use_optimized:
            final_emb = self._forward_opt(x, cluster_ids)
        else:
            final_emb = self._forward(x, cluster_ids)
        logits = F.linear(final_emb, self.get_output_projection_weights())
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_embedding_parameters(self):
        count = sum(p.numel() for p in self.embedding.parameters())
        if self.config.chunk_embeddings:
            count += sum(p.numel() for p in self.chunk_value_embedding.parameters())
        return count

    def num_non_embedding_parameters(self):
        return self.num_parameters() - self.num_embedding_parameters()
