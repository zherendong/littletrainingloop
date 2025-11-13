"""
Weight initialization utilities for neural networks.

This module provides initialization schemes tailored for different activation functions,
especially for SiLU/Swish and GLU variants which require special consideration.
"""

import math
import torch
import torch.nn as nn


def calculate_silu_gain() -> float:
    """
    Calculate the gain factor for SiLU activation to preserve variance.

    For SiLU(x) = x * sigmoid(x), when x ~ N(0, σ²), numerical analysis shows:
    E[SiLU²(x)] ≈ 0.355 * σ²

    To preserve variance (make E[SiLU²(x)] = σ²), we need to scale by:
    gain = 1 / sqrt(0.355) ≈ 1.679

    Reference: Kolmogorov-Arnold Transformer paper (arXiv:2409.10594v1)
    """
    # Empirically determined variance scaling factor for SiLU
    silu_variance_factor = 0.355
    return 1.0 / math.sqrt(silu_variance_factor)


def calculate_gelu_gain() -> float:
    """
    Calculate the gain factor for GELU activation to preserve variance.

    For GELU, the variance scaling is approximately 0.5.
    """
    gelu_variance_factor = 0.5
    return 1.0 / math.sqrt(gelu_variance_factor)


def init_linear(
    linear: nn.Linear,
    activation: str = "linear",
    mode: str = "fan_in",
    scaling_factor: float = 1.0,
    pairwise_mode: str | None = None,
) -> None:
    """
    Initialize a linear layer's weight with variance-preserving initialization.

    Args:
        weight: Weight tensor to initialize (out_features, in_features)
        activation: Type of activation function following this layer.
                   One of: "linear", "relu", "silu", "swish", "gelu"
        mode: Either "fan_in" (default) or "fan_out"
    """
    weight = linear.weight
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    fan = fan_in if mode == "fan_in" else fan_out

    # Determine gain based on activation
    if activation in ["linear", "none"]:
        gain = 1.0
    elif activation == "relu":
        gain = math.sqrt(2.0)  # Standard Kaiming initialization
    elif activation in ["silu", "swish"]:
        gain = calculate_silu_gain()
    elif activation == "gelu":
        gain = calculate_gelu_gain()
    elif activation == "polyrelu":
        gain = 0.8460  # determined with ChatGPT
    elif activation == "polynorm":
        gain = 1.0
    elif activation == "segmented":
        gain = 1.0
    else:
        raise ValueError(f"Unknown activation: {activation}")

    std = gain * scaling_factor / math.sqrt(fan)
    with torch.no_grad():
        weight.normal_(0, std)

        if pairwise_mode is not None:
            assert pairwise_mode in ["equal", "opposing"]
            print("Applying pairwise initialization")
            if pairwise_mode == "equal":
                weight[1::2, :] = weight[::2, :]
            elif pairwise_mode == "opposing":  # note that we flip the dims
                weight[:, 1::2] = -weight[:, ::2]


def init_embedding(
    embedding: nn.Embedding,
    scaling_factor: float = 1.0,
) -> None:
    """
    Initialize embedding layer.

    Args:
        embedding: Embedding layer to initialize
        std: Standard deviation. If None, uses 1/sqrt(embedding_dim)
    """
    std = scaling_factor / math.sqrt(embedding.embedding_dim)

    with torch.no_grad():
        embedding.weight.normal_(0, std)


def print_initialization_info(
    layer_name: str,
    weight: torch.Tensor,
    activation: str = "linear",
) -> None:
    """
    Print diagnostic information about weight initialization.

    Useful for debugging initialization issues.
    """
    with torch.no_grad():
        std = torch.std(weight).item()
        mean = torch.mean(weight).item()
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)

        print(f"{layer_name}:")
        print(f"  Shape: {weight.shape}")
        print(f"  Fan in/out: {fan_in}/{fan_out}")
        print(f"  Actual std: {std:.6f}")
        print(f"  Actual mean: {mean:.6f}")

        if activation in ["silu", "swish"]:
            expected_std = calculate_silu_gain() / math.sqrt(fan_in)
            print(f"  Expected std (SiLU): {expected_std:.6f}")
        elif activation == "relu":
            expected_std = math.sqrt(2.0) / math.sqrt(fan_in)
            print(f"  Expected std (ReLU): {expected_std:.6f}")
        else:
            expected_std = 1.0 / math.sqrt(fan_in)
            print(f"  Expected std (linear): {expected_std:.6f}")
