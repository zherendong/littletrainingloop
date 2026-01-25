# Little Training Loop

PyTorch framework for training transformer language models with a focus on scaling experiments and architectural exploration.

## Project Structure

```
├── transformer.py          # Core transformer model (MLP, Attention, TransformerBlock)
├── language_model_training.py  # Main training entry point
├── training_loop.py        # Generic training loop
├── training_basics.py      # Config dataclasses, TrainingState
├── run_scaling_series.py   # Orchestrate scaling experiments across GPUs
├── model_configs/          # Predefined model architectures (chinchilla sizes)
├── experiments/            # Self-contained experiments (each may have its own claude.md)
├── scaling_analysis/       # Scaling law analysis and visualization
├── data/                   # Dataset directories (slimpajama, stackv2)
└── *_test.py              # Tests for each module
```

## Running Training

### Single Training Run
```bash
python language_model_training.py --model_config chinchilla-44m --no_neptune --description "test run"
```

### Scaling Experiments
`run_scaling_series.py` is the main tool for running training experiments at scale:
```bash
# Single GPU with specific config variants
python run_scaling_series.py --neptune_tags experiment_name

# Multi-GPU across machines (dispatches to tmux sessions)
python run_scaling_series.py --multi_gpu --neptune_tags experiment_name

# Check job status across machines
python run_scaling_series.py --check_jobs

# Stop all running jobs
python run_scaling_series.py --stop_all_jobs
```

### Running Tests
```bash
pytest transformer_test.py
pytest -v  # run all tests
```

## Key Patterns

### Configuration
- Use frozen dataclasses for configs (`TransformerConfig`, `TrainingConfig`, `LanguageModelTrainingConfig`)
- Model configs registered via `ConfigRegistry` decorator in `model_configs/`
- Modify configs with `dataclasses.replace()` for variants

### Model Architecture
- `TransformerConfig` controls all architecture options:
  - `gqa`, `glu`: Standard architectural choices
  - `nonlinearity`: "swish", "gelu", "relu", "PolyReLU", "PolyNorm"
  - `spelling_bee`: Character-level embedding experiments
  - `pre_projection_transform`: Query projection variants
  - `zheren_init`, `pairwise_cancelling_init`: Initialization schemes

### Training
- BFloat16 precision by default
- `torch.compile` with inductor caching enabled
- Neptune for experiment tracking (disable with `--no_neptune`)
- Checkpointing via `checkpointing.py`

### Data Loading
- Multiprocess data loading (`multiprocess_iterable.py`)
- Datasets: SlimPajama, StackV2, Strawberry
- Shuffle buffer for randomization

## Code Style

- Tests named `*_test.py`, use pytest
- Type hints throughout
- Docstrings for public functions
- No TensorFlow imports (blocked at module level)

### File Naming
Never create generic "utility" files like `utils.py`, `helpers.py`, or `common.py`. Instead, name files after what they actually do:
- `embedder.py` instead of `utils.py` (for embedding functionality)
- `sampling.py` instead of `helpers.py` (for data sampling)
- `topk.py` instead of `search_utils.py` (for top-k search)

This makes it clear what each file contains and avoids dumping unrelated functionality into catch-all files.

### Import Convention
Prefer importing the module, not individual classes/functions:

```python
# Preferred
import transformer

model = transformer.TransformerModel(config)
block = transformer.TransformerBlock(...)

# Avoid
from transformer import TransformerModel, TransformerBlock

model = TransformerModel(config)
```

This makes it clear where classes and functions come from and avoids namespace pollution.

### Top-level Imports Only
All imports should be at the top of the file. Avoid inline/lazy imports inside functions or methods:

```python
# Preferred - at top of file
import hashlib
import os

def my_function():
    fingerprint = hashlib.sha256(data).hexdigest()

# Avoid - inside function
def my_function():
    import hashlib  # Don't do this
    import os
    fingerprint = hashlib.sha256(data).hexdigest()
```

This makes dependencies explicit and avoids hidden import costs during execution.

## Dependencies

Requires CUDA. Key packages:
- PyTorch with CUDA (cu128)
- flash-attn
- torchtune, lm_eval
- neptune-client
- tiktoken

Install with:
```bash
pip install -r requirements.txt
```


## Common Tasks

### Adding a New Model Config
Add to `model_configs/chinchilla.py` using the registry:
```python
@transformer.config_registry.register("my-model")
def my_model():
    return TransformerConfig(num_layers=12, ...)
```

### Running Evaluation
```bash
python eval_main.py --checkpoint_path <path>
```

### Creating Experiment Variants
In `run_scaling_series.py`, modify `config_variants()` to generate experiment sweeps.


## Experiments
Experiments are in `experiments/` and have their own claude.md and/or spec.md files. Before we touch the main codebase with changes we should try out the changes in the experiments directory. This is meant to
- reduce the explosion of options in the main codebase, thereby maintaining a high throughput of experiments,
- provide a way to refine the plan before we implement it, and
- encourage finding minimal and fast implementations for ideas before we scale them up.
