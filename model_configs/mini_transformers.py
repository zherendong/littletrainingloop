import transformer


@transformer.transformer_config_registry.register("mini-transformer-1")
def mini_transformer_1() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=2,
        num_heads=1,
        num_heads_kv=1,
        head_dim=16,
        mlp_inner_size=64,
        embedding_size=16,
        use_flash_attention=False,
    )


@transformer.transformer_config_registry.register("mini-transformer-2")
def mini_transformer_2() -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        num_layers=2,
        num_heads=2,
        num_heads_kv=2,
        head_dim=16,
        mlp_inner_size=128,
        embedding_size=32,
        use_flash_attention=False,
    )
