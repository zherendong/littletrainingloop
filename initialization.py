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


def init_linear_weight(
    weight: torch.Tensor,
    activation: str = "linear",
    mode: str = "fan_in",
) -> None:
    """
    Initialize a linear layer's weight with variance-preserving initialization.
    
    Args:
        weight: Weight tensor to initialize (out_features, in_features)
        activation: Type of activation function following this layer.
                   One of: "linear", "relu", "silu", "swish", "gelu"
        mode: Either "fan_in" (default) or "fan_out"
    """
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
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    # Calculate standard deviation
    std = gain / math.sqrt(fan)
    
    # Initialize with normal distribution
    with torch.no_grad():
        weight.normal_(0, std)


def init_swiglu_weights(
    linear_in: nn.Linear,
    linear_glu: nn.Linear,
    input_size: int,
) -> None:
    """
    Initialize weights for SwiGLU (Gated Linear Unit with SiLU activation).
    
    For SwiGLU: output = SiLU(W1 @ x) * (W2 @ x)
    
    Both W1 and W2 should be initialized with SiLU-aware initialization.
    The gating mechanism (multiplication) approximately preserves variance,
    so we use the same initialization for both.
    
    Args:
        linear_in: The first linear layer (W1)
        linear_glu: The gating linear layer (W2)
        input_size: Input dimension
    """
    # Initialize both with SiLU-aware initialization
    init_linear_weight(linear_in.weight, activation="silu", mode="fan_in")
    init_linear_weight(linear_glu.weight, activation="linear", mode="fan_in")


def init_output_projection(
    linear: nn.Linear,
    depth: int | None = None,
    use_depth_scaling: bool = False,
) -> None:
    """
    Initialize output projection layer, optionally with depth scaling.
    
    Depth scaling helps with training stability in deep networks by scaling
    the initialization variance by 1/sqrt(2*depth), as suggested in the
    GPT-2/GPT-3 papers and muP (maximal update parametrization).
    
    Args:
        linear: Output projection linear layer
        depth: Number of layers in the network (for depth scaling)
        use_depth_scaling: Whether to apply depth-dependent scaling
    """
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    
    # Base standard deviation (variance-preserving for linear activation)
    std = 1.0 / math.sqrt(fan_in)
    
    # Apply depth scaling if requested
    if use_depth_scaling and depth is not None:
        # Scale by 1/sqrt(2*depth) as in GPT-2/GPT-3
        # The factor of 2 accounts for both attention and MLP residual branches
        std = std / math.sqrt(2.0 * depth)
    
    with torch.no_grad():
        linear.weight.normal_(0, std)


def init_embedding(
    embedding: nn.Embedding,
    std: float | None = None,
) -> None:
    """
    Initialize embedding layer.
    
    Args:
        embedding: Embedding layer to initialize
        std: Standard deviation. If None, uses 1/sqrt(embedding_dim)
    """
    if std is None:
        std = 1.0 / math.sqrt(embedding.embedding_dim)
    
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


# Convenience functions for common patterns

def init_mlp_with_silu(
    linear_in: nn.Linear,
    linear_out: nn.Linear,
    depth: int | None = None,
    use_depth_scaling: bool = False,
) -> None:
    """Initialize MLP with SiLU activation: linear_out(SiLU(linear_in(x)))"""
    init_linear_weight(linear_in.weight, activation="silu")
    init_output_projection(linear_out, depth=depth, use_depth_scaling=use_depth_scaling)


def init_mlp_with_relu(
    linear_in: nn.Linear,
    linear_out: nn.Linear,
    depth: int | None = None,
    use_depth_scaling: bool = False,
) -> None:
    """Initialize MLP with ReLU activation: linear_out(ReLU(linear_in(x)))"""
    init_linear_weight(linear_in.weight, activation="relu")
    init_output_projection(linear_out, depth=depth, use_depth_scaling=use_depth_scaling)


def init_mlp_with_gelu(
    linear_in: nn.Linear,
    linear_out: nn.Linear,
    depth: int | None = None,
    use_depth_scaling: bool = False,
) -> None:
    """Initialize MLP with GELU activation: linear_out(GELU(linear_in(x)))"""
    init_linear_weight(linear_in.weight, activation="gelu")
    init_output_projection(linear_out, depth=depth, use_depth_scaling=use_depth_scaling)

