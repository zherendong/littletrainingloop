# Proper Initialization for SiLU/SwiGLU Networks

## Problem

PyTorch's default `nn.Linear()` initialization uses Kaiming uniform initialization, which is designed for ReLU activations with a gain factor of √2. When using SiLU (Swish) activation, this causes **variance reduction** because:

- For ReLU: E[ReLU²(x)] ≈ 0.5σ² (hence gain = √2)
- For SiLU: E[SiLU²(x)] ≈ 0.355σ² (requires gain ≈ 1.679)

This variance reduction can lead to vanishing activations in deep networks and poor training performance.

## Solution

We've implemented activation-aware initialization that preserves variance through different activation functions:

1. **SiLU-aware initialization**: Uses gain factor of 1.679 (1/√0.355)
2. **SwiGLU-aware initialization**: Properly initializes both gates in gated linear units
3. **Depth-scaled initialization**: Optional scaling for output projections in deep networks (GPT-2/GPT-3 style)

## Usage

### Basic Configuration

Enable proper initialization in your `TransformerConfig`:

```python
config = TransformerConfig(
    num_layers=12,
    num_heads=8,
    head_dim=64,
    embedding_size=512,
    glu=True,                    # Use SwiGLU
    nonlinearity="swish",        # SiLU activation
    use_proper_init=True,        # Enable activation-aware initialization
    use_depth_scaling=False,     # Optional: enable depth scaling
)
```

### Configuration Options

- **`use_proper_init`** (default: `True`): Enable activation-aware initialization
  - When `True`: Uses variance-preserving initialization for SiLU, GELU, ReLU
  - When `False`: Uses PyTorch's default Kaiming uniform initialization

- **`use_depth_scaling`** (default: `False`): Enable depth-dependent scaling for output projections
  - When `True`: Scales output projection weights by 1/√(2*depth)
  - Recommended for very deep networks (>24 layers)
  - Based on GPT-2/GPT-3 initialization strategy

### Initialization Details

#### For MLP Layers

**Standard MLP (no GLU):**
- Input projection: Initialized with activation-specific gain
  - ReLU: gain = √2
  - GELU: gain ≈ 1.702
  - SiLU: gain ≈ 1.679
- Output projection: Standard initialization with optional depth scaling

**SwiGLU (GLU=True):**
- Gate 1 (linear_in): Initialized with SiLU gain (1.679)
- Gate 2 (linear_glu): Initialized with linear gain (1.0)
- Output projection: Standard initialization with optional depth scaling

#### For Attention Layers

- Q, K, V projections: Linear initialization (gain = 1.0)
- Output projection: Standard initialization with optional depth scaling

#### For Embeddings

- Embedding layer: Normal initialization with std = 1/√(embedding_dim)
- Output projection: Standard initialization (no depth scaling)

## Testing

Run the initialization tests to verify variance preservation:

```bash
python test_initialization.py
```

This will test:
1. Variance preservation through SiLU activation
2. Variance preservation through SwiGLU
3. Depth scaling behavior
4. Full transformer model with proper initialization
5. Comparison of different initialization schemes

### Expected Results

With proper initialization, you should see:
- **SiLU variance ratio**: ~0.96-1.04 (close to 1.0)
- **SwiGLU variance ratio**: ~1.0-1.2 (within 20% of 1.0)
- **Default PyTorch variance ratio**: ~0.03-0.09 (severe variance reduction!)

## Implementation Details

The initialization logic is in `initialization.py`:

- `calculate_silu_gain()`: Returns 1.679 for SiLU
- `calculate_gelu_gain()`: Returns 1.702 for GELU
- `init_linear_weight()`: General weight initialization with activation-aware gains
- `init_swiglu_weights()`: Specialized initialization for SwiGLU layers
- `init_output_projection()`: Output projection with optional depth scaling
- `init_embedding()`: Embedding layer initialization

## Migration Guide

### Existing Models

If you have existing models trained with default initialization, you can:

1. **Start fresh**: Set `use_proper_init=True` for new training runs
2. **Compare**: Run experiments with both `use_proper_init=True` and `False` to measure impact
3. **Gradual adoption**: Enable for new model variants while keeping old configs unchanged

### Backward Compatibility

The default is `use_proper_init=True`, but existing configs without this field will use the new initialization. To maintain exact backward compatibility with old models:

```python
config = TransformerConfig(
    # ... other settings ...
    use_proper_init=False,  # Use old PyTorch default initialization
)
```

## References

- **Kolmogorov-Arnold Transformer** (arXiv:2409.10594v1): Analyzes variance scaling for SiLU
- **GPT-2/GPT-3 papers**: Depth-scaled initialization for residual connections
- **Kaiming He et al.**: Original Kaiming initialization for ReLU networks

## Recommendations

1. **For SwiGLU models**: Always use `use_proper_init=True`
2. **For deep models (>24 layers)**: Consider `use_depth_scaling=True`
3. **For shallow models (<12 layers)**: `use_depth_scaling=False` is fine
4. **For ReLU/GELU**: Proper init still helps, but the difference is smaller

## Troubleshooting

### Variance still not preserved?

Check:
- Are you using the correct activation function in config?
- Is `use_proper_init=True` in your config?
- Are you using custom layers (SkinnyLinear, CopyLinear) that may need updates?

### Training unstable with depth scaling?

Try:
- Reduce learning rate
- Disable depth scaling (`use_depth_scaling=False`)
- Use gradient clipping

### NaN values during training?

This is likely unrelated to initialization. Check:
- Learning rate (may be too high)
- Gradient clipping
- Mixed precision settings
- Data preprocessing

