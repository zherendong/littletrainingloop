# Minimal Training Loop

A simple PyTorch training loop implementation with a linear model and random data.

## Features

- **Linear Model**: Simple neural network with one linear layer (y = Wx + b)
- **Training Configuration**: Clean configuration object for all hyperparameters
- **Random Data Generation**: Creates synthetic data with a known linear relationship
- **Training Loop**: Complete training implementation with loss tracking
- **Loss Summary**: Detailed loss analysis and progress tracking
- **Weight Comparison**: Compares learned weights with true weights

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the minimal training loop:

```bash
# Using Python 3.11 directly
/opt/homebrew/bin/python3.11 minimal_training_loop.py

# Or using the convenience script
chmod +x run.sh
./run.sh minimal_training_loop.py

# Or using the entry point
/opt/homebrew/bin/python3.11 start.py
```

## What it does

1. **Data Generation**: Creates random input data (X) and corresponding outputs (y) with a known linear relationship
2. **Model Creation**: Initializes a simple linear model with configurable input/output dimensions
3. **Training**: Runs a training loop using SGD optimizer and MSE loss
4. **Monitoring**: Tracks and displays loss during training
5. **Evaluation**: Compares learned parameters with the true parameters used to generate data
6. **Visualization**: Shows training loss curve

## Customization

You can modify the hyperparameters in the `main()` function:

- `input_size`: Dimension of input features
- `output_size`: Dimension of output
- `num_samples`: Number of training samples
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for SGD optimizer

## Example Output

```
=== Minimal PyTorch Training Loop ===
Input size: 10
Output size: 1
Number of samples: 1000
Number of epochs: 100

Generating random data...
Data shapes - X: torch.Size([1000, 10]), y: torch.Size([1000, 1])

Creating linear model...
Model: LinearModel(
  (linear): Linear(in_features=10, out_features=1, bias=True)
)
Number of parameters: 11

Starting training for 100 epochs...
Learning rate: 0.01
--------------------------------------------------
Epoch [10/100], Loss: 0.234567
Epoch [20/100], Loss: 0.123456
...
```
