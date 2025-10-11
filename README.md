# Language Model Training

A PyTorch implementation for training transformer language models on large text datasets like SlimPajama.

## Features

- **Transformer Model**: Transformer architecture with configurable parameters
- **Training Configuration**: Clean configuration object for all hyperparameters
- **SlimPajama Dataset**: Support for the SlimPajama-627B dataset with efficient data loading
- **Training Loop**: Complete training implementation with loss tracking and evaluation
- **Multi-process Data Loading**: Efficient data loading with separate processes
- **Neptune Integration**: Optional experiment tracking with Neptune
- **Chinchilla Scaling**: Automatic computation of optimal numer of training steps

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

First, download and prepare the SlimPajama dataset:

```bash
# Download training data
python download_data.py --dataset slimpajama --split train

# Download validation data  
python download_data.py --dataset slimpajama --split validation
```

This will create `data/slimpajama_train/` and `data/slimpajama_validation/` directories with the processed JSONL files.

## Usage

Run the language model training:

```bash
# Training without Neptune logging
python language_model_training.py --no_neptune --description "Local training"

# Use different model configuration
python language_model_training.py --model_config chinchilla-44m --description "Small model test"

# Profile mode (short run for testing)
python language_model_training.py --profile_only
```

## What it does

1. **Data Loading**: Loads SlimPajama dataset with tokenization and batching
2. **Model Creation**: Initializes a transformer model with specified configuration
3. **Training**: Runs training loop with AdamW optimizer and learning rate scheduling
4. **Monitoring**: Tracks loss, learning rate, and performance metrics during training
5. **Evaluation**: Periodic evaluation on validation data
6. **Experiment Tracking**: Optional Neptune integration for experiment management

## Customization

You can modify the hyperparameters in the `run()` function in `language_model_training.py`:

- `batch_size`: Training batch size
- `sequence_length`: Maximum sequence length
- `learning_rate`: Learning rate for AdamW optimizer
- `warmup_steps`: Number of warmup steps for learning rate schedule
- `model_config`: Transformer architecture configuration
