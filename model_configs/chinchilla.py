"""Chinchilla model configs.

Taken from the Chinchilla paper.

https://arxiv.org/pdf/2203.15556#page=36

Warning: the parameter counts are off. It appears the chinchilla parameter counts
include the output projection parameters. Their layers have a weirdly low parameter
count of ~13.75M for a layer with embedding size 1024.
"""

import transformer

registry = transformer.transformer_config_registry


@registry.register("chinchilla-44m")
def chinchilla_44m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=8,
        num_heads=8,
        head_dim=64,
        mlp_inner_size=2048,
        embedding_size=512,
    )


@registry.register("chinchilla-57m")
def chinchilla_57m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=9,
        num_heads=9,
        head_dim=64,
        mlp_inner_size=2304,
        embedding_size=576,
    )


@registry.register("chinchilla-74m")
def chinchilla_74m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=10,
        num_heads=10,
        head_dim=64,
        mlp_inner_size=2560,
        embedding_size=640,
    )


@registry.register("chinchilla-90m")
def chinchilla_90m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=13,
        num_heads=10,
        head_dim=64,
        mlp_inner_size=2560,
        embedding_size=640,
    )


@registry.register("chinchilla-106m")
def chinchilla_106m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=16,
        num_heads=10,
        head_dim=64,
        mlp_inner_size=2560,
        embedding_size=640,
    )


@registry.register("chinchilla-117m")
def chinchilla_117m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=12,
        num_heads=12,
        head_dim=64,
        mlp_inner_size=3072,
        embedding_size=768,
        use_proper_init=True,        # Enable activation-aware initialization
        use_depth_scaling=False,     # Optional: enable depth scaling
    )


@registry.register("chinchilla-140m")
def chinchilla_140m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=15,
        num_heads=12,
        head_dim=64,
        mlp_inner_size=3072,
        embedding_size=768,
    )


@registry.register("chinchilla-163m")
def chinchilla_163m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=18,
        num_heads=12,
        head_dim=64,
        mlp_inner_size=3072,
        embedding_size=768,
    )


@registry.register("chinchilla-175m")
def chinchilla_175m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=14,
        num_heads=14,
        head_dim=64,
        mlp_inner_size=3584,
        embedding_size=896,
    )


@registry.register("chinchilla-196m")
def chinchilla_196m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=16,
        num_heads=14,
        head_dim=64,
        mlp_inner_size=3584,
        embedding_size=896,
    )


@registry.register("chinchilla-217m")
def chinchilla_217m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=18,
        num_heads=14,
        head_dim=64,
        mlp_inner_size=3584,
        embedding_size=896,
    )


@registry.register("chinchilla-251m")
def chinchilla_251m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=16,
        num_heads=16,
        head_dim=64,
        mlp_inner_size=4096,
        embedding_size=1024,
    )


@registry.register("chinchilla-278m")
def chinchilla_278m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=18,
        num_heads=16,
        head_dim=64,
        mlp_inner_size=4096,
        embedding_size=1024,
    )


@registry.register("chinchilla-306m")
def chinchilla_306m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=20,
        num_heads=16,
        head_dim=64,
        mlp_inner_size=4096,
        embedding_size=1024,
    )


@registry.register("chinchilla-425m")
def chinchilla_425m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=18,
        num_heads=10,
        head_dim=128,
        mlp_inner_size=5120,
        embedding_size=1280,
    )


@registry.register("chinchilla-489m")
def chinchilla_489m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=21,
        num_heads=10,
        head_dim=128,
        mlp_inner_size=5120,
        embedding_size=1280,
    )


@registry.register("chinchilla-509m")
def chinchilla_509m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=18,
        num_heads=11,
        head_dim=128,
        mlp_inner_size=5632,
        embedding_size=1408,
    )


@registry.register("chinchilla-552m")
def chinchilla_552m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=24,
        num_heads=10,
        head_dim=128,
        mlp_inner_size=5120,
        embedding_size=1280,
    )


@registry.register("chinchilla-587m")
def chinchilla_587m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=21,
        num_heads=11,
        head_dim=128,
        mlp_inner_size=5632,
        embedding_size=1408,
    )


@registry.register("chinchilla-632m")
def chinchilla_632m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=19,
        num_heads=12,
        head_dim=128,
        mlp_inner_size=6144,
        embedding_size=1536,
    )


@registry.register("chinchilla-724m")
def chinchilla_724m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=22,
        num_heads=12,
        head_dim=128,
        mlp_inner_size=6144,
        embedding_size=1536,
    )


@registry.register("chinchilla-816m")
def chinchilla_816m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=25,
        num_heads=12,
        head_dim=128,
        mlp_inner_size=6144,
        embedding_size=1536,
    )


@registry.register("chinchilla-1266m")
def chinchilla_1266m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=22,
        num_heads=16,
        head_dim=128,
        mlp_inner_size=8192,
        embedding_size=2048,
    )


@registry.register("chinchilla-1593m")
def chinchilla_1593m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=28,
        num_heads=16,
        head_dim=128,
        mlp_inner_size=8192,
        embedding_size=2048,
    )


@registry.register("chinchilla-2298m")
def chinchilla_2298m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=26,
        num_heads=20,
        head_dim=128,
        mlp_inner_size=10240,
        embedding_size=2560,
    )


@registry.register("chinchilla-4516m")
def chinchilla_4516m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=36,
        num_heads=24,
        head_dim=128,
        mlp_inner_size=12288,
        embedding_size=3072,
    )


@registry.register("chinchilla-9293m")
def chinchilla_9293m() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=42,
        num_heads=32,
        head_dim=128,
        mlp_inner_size=16384,
        embedding_size=4096,
    )


# Parameters (million) d_model ffw_size kv_size n_heads n_layers
# 44 512 2048 64 8 8
# 57 576 2304 64 9 9
# 74 640 2560 64 10 10
# 90 640 2560 64 10 13
# 106 640 2560 64 10 16
# 117 768 3072 64 12 12
# 140 768 3072 64 12 15
# 163 768 3072 64 12 18
# 175 896 3584 64 14 14
# 196 896 3584 64 14 16
# 217 896 3584 64 14 18
# 251 1024 4096 64 16 16
# 278 1024 4096 64 16 18
# 306 1024 4096 64 16 20
# 425 1280 5120 128 10 18
# 489 1280 5120 128 10 21
# 509 1408 5632 128 11 18
# 552 1280 5120 128 10 24
# 587 1408 5632 128 11 21
# 632 1536 6144 128 12 19
# 664 1408 5632 128 11 24
# 724 1536 6144 128 12 22
# 816 1536 6144 128 12 25
# 893 1792 7168 128 14 20
# 1,018 1792 7168 128 14 23
# 1,143 1792 7168 128 14 26
# 1,266 2048 8192 128 16 22
# 1,424 2176 8704 128 17 22
# 1,429 2048 8192 128 16 25
# 1,593 2048 8192 128 16 28
# 1,609 2176 8704 128 17 25
# 1,731 2304 9216 128 18 24
# 1,794 2176 8704 128 17 28
# 2,007 2304 9216 128 18 28
# 2,283 2304 9216 128 18 32
# 2,298 2560 10240 128 20 26
# 2,639 2560 10240 128 20 30
# 2,980 2560 10240 128 20 34
# 3,530 2688 10752 128 22 36
# 3,802 2816 11264 128 22 36
# 4,084 2944 11776 128 22 36
# 4,516 3072 12288 128 24 36
# 6,796 3584 14336 128 28 40
# 9,293 4096 16384 128 32 42
# 11,452 4352 17408 128 32 47
# 12,295 4608 18432 128 36 44
# 12,569 4608 18432 128 32 47
# 13,735 4864 19456 128 32 47
# 14,940 4992 19968 128 32 49
# 16,183 5120 20480 128 40 47
