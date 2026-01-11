# Extending MLPs During Training

## Core Concept

The goal is to dynamically extend MLPs in a transformer during training by adding more key-value (KV) pairs.

### MLP as Key-Value Store

In a standard MLP with GLU:
```
x_norm = norm(x)
keys = linear_in(x_norm)        # shape: (batch, seq, inner_dim) 
gates = linear_glu(x_norm)      # shape: (batch, seq, inner_dim)
activations = nonlinear(keys) * gates  # which KV pairs are "active"
output = linear_out(activations)  # values projected back
```

- **Keys**: First projection weights (`linear_in.weight`, shape: `inner_dim x input_size`)
- **Values**: Output projection weights (`linear_out.weight`, shape: `output_size x inner_dim`)
- **Activations**: The intermediate tensor tells us how much each KV pair contributes

### Extension Strategy

1. **Block size**: 512 KV pairs per block (configurable)
2. **Wrapper module**: `GrowingMLP` manages a list of sub-MLPs
3. **Growth**: Add new blocks over time during training
4. **Learning rate schedule**: New blocks need their own warmup

## Current MLP Architecture (from transformer.py)

```python
class MLP(nn.Module):
    def __init__(self, *, dtype, input_size, glu=False, ...):
        self.inner_size = input_size * 4  # typically
        self.linear_in = nn.Linear(input_size, inner_size, bias=False)
        if glu:
            self.linear_glu = nn.Linear(input_size, inner_size, bias=False)
        self.linear_out = nn.Linear(inner_size, output_size, bias=False)
```

## Design Questions

### 1. Module Structure

How should the wrapper work?

**Option A**: Wrapper holds a list of MLP sub-modules
```python
class GrowingMLP(nn.Module):
    def __init__(self, config, block_size=512):
        self.blocks = nn.ModuleList([MLPBlock(config, block_size)])
    
    def forward(self, x):
        return sum(block(x) for block in self.blocks)
```

**Option B**: Single MLP with dynamically resized tensors
- More complex, but potentially more efficient

### 2. Learning Rate Per Block

How to give each block its own LR schedule?

**Option A**: Parameter groups in optimizer
```python
optimizer = AdamW([
    {'params': block0.parameters(), 'lr': lr},
    {'params': block1.parameters(), 'lr': lr * 0.1},  # new block, still warming up
])
```

**Option B**: Custom scheduler that tracks block "age"

**Option C**: Use gradient scaling instead of LR adjustment

### 3. When to Add New Blocks

- Fixed schedule (every N steps)?
- Based on some metric (loss plateau, gradient norms)?
- Manual trigger?

### 4. Initialization of New Blocks

- Zero-init so new blocks start with no contribution?
- Small random init?
- Copy from existing blocks with perturbation?

## Design Decisions

1. **Module structure**: Option A - wrapper with list of sub-MLPs (simpler)
2. **Norm layer**: Each block has its own norm (cleaner learning dynamics - new blocks don't depend on how existing norms have evolved)
3. **Learning rate**: Fresh warmup for each new block (independent schedule)
4. **Block addition**: At specific training steps defined upfront
5. **Initialization**: Use `pairwise_cancelling_init` for new blocks
6. **GLU**: Each block has its own GLU weights

## Implementation Notes

### Fresh Warmup Strategy

Since PyTorch schedulers operate on all param groups, we'll use a **LR multiplier** approach:
- Each param group has a `warmup_start_step` and `warmup_steps` attribute
- A custom scheduler wrapper computes per-group multipliers
- `effective_lr = base_lr * warmup_multiplier(current_step - warmup_start_step)`

### MLPBlock Structure

Each block is a self-contained MLP slice with its own norm:
```python
class MLPBlock(nn.Module):
    # Has: norm, linear_in, linear_glu (if GLU), nonlinear_fn, linear_out
    # inner_size = block_size (e.g., 512)
    def forward(self, x):
        x_norm = self.norm(x)  # each block normalizes independently
        ...
```

### GrowingMLP Structure

```python
class GrowingMLP(nn.Module):
    def __init__(self, config, block_size=512):
        self.blocks = nn.ModuleList([MLPBlock(...)])

    def forward(self, x):
        return sum(block(x) for block in self.blocks)  # each block norms internally

    def add_block(self):
        new_block = MLPBlock(...)
        new_block.init_weights()  # pairwise_cancelling_init
        self.blocks.append(new_block)
        return new_block.parameters()  # for optimizer
```

## Open Questions

1. How does this interact with gradient checkpointing in TransformerBlock?
2. How do we handle saving/loading checkpoints with variable-size models?
3. Should we support "pruning" (removing low-contribution blocks)?

## Files

- `growing_mlp.py` - `MLPBlock` and `GrowingMLP` classes
- `block_scheduler.py` - `BlockAwareScheduler` for per-block warmup
- `example.py` - Simple tests and training example

## Status

- [x] Create `MLPBlock` class
- [x] Create `GrowingMLP` wrapper
- [x] Create warmup-aware LR scheduler
- [x] Write simple test/example
- [ ] Integrate with training loop (use in `TransformerBlock`)
- [ ] Handle gradient checkpointing
- [ ] Checkpoint save/load with variable blocks

## Remaining Questions

1. **Integration with TransformerBlock**: Replace `self.mlp` with `GrowingMLP`? Or wrap it?
2. **Training state integration**: The `LanguageModelTrainingState` needs to know when to add blocks and how to update the optimizer/scheduler.
3. **Config changes**: Need to add block addition schedule to `TransformerConfig` or `TrainingConfig`?

