import abc

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters"""

    input_size: int = 10
    output_size: int = 3
    num_samples: int = 1000
    num_epochs: int = 100
    learning_rate: float = 0.1


class TrainingState(abc.ABC):
    pass


class LinearModel(nn.Module):
    """Simple linear model: y = Wx + b"""

    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def generate_random_data(config: TrainingConfig):
    """Generate random input-output pairs using configuration"""
    X = torch.randn(config.num_samples, config.input_size)
    # Create a simple linear relationship with some noise
    true_weights = torch.randn(config.input_size, config.output_size)
    true_bias = torch.randn(config.output_size)
    y = (
        X @ true_weights
        + true_bias
        + 0.1 * torch.randn(config.num_samples, config.output_size)
    )
    return X, y, true_weights, true_bias


def train_model(
    model: LinearModel,
    model_inputs: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig,
):
    """Training loop using configuration object"""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    losses = []

    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Learning rate: {config.learning_rate}")
    print("-" * 50)

    for epoch in range(config.num_epochs):
        # Forward pass
        predictions = model(model_inputs)
        loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss for plotting
        loss_numpy = loss.detach().numpy()
        losses.append(loss_numpy)

        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item():.4f}")

    print("-" * 50)
    print(f"Training completed! Final loss: {losses[-1]:.4f}")
    return losses


def print_loss_summary(losses):
    """Print a summary of the training loss"""
    print(f"\nLoss Summary:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")

    # Print some intermediate values
    if len(losses) >= 10:
        quarter = len(losses) // 4
        print(f"Loss at 25%: {losses[quarter]:.4f}")
        print(f"Loss at 50%: {losses[2 * quarter]:.4f}")
        print(f"Loss at 75%: {losses[3 * quarter]:.4f}")


def compare_weights(model, true_weights, true_bias):
    """Compare learned weights with true weights"""
    learned_weights = model.linear.weight.data
    learned_bias = model.linear.bias.data

    print("\nWeight Comparison:")
    print(f"True weights shape: {true_weights.shape}")
    print(f"Learned weights shape: {learned_weights.shape}")
    print(
        f"Weight difference (L2 norm): {torch.norm(learned_weights.T - true_weights).item():.6f}"
    )

    print(f"\nBias Comparison:")
    print(f"True bias: {true_bias.numpy()}")
    print(f"Learned bias: {learned_bias.numpy()}")
    print(
        f"Bias difference (L2 norm): {torch.norm(learned_bias - true_bias).item():.6f}"
    )


def run():
    # Create training configuration
    config = TrainingConfig()

    print("=== Minimal PyTorch Training Loop ===")
    print(f"Input size: {config.input_size}")
    print(f"Output size: {config.output_size}")
    print(f"Number of samples: {config.num_samples}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print()

    # Generate random data
    print("Generating random data...")
    X, y, true_weights, true_bias = generate_random_data(config)
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print()

    # Create model
    print("Creating linear model...")
    model = LinearModel(config.input_size, config.output_size)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print()

    # Train the model
    losses = train_model(model, X, y, config)

    # Compare learned weights with true weights
    compare_weights(model, true_weights, true_bias)

    # Print training loss summary
    print_loss_summary(losses)

    # Test the model with new data
    print("\nTesting with new data...")
    X_test = torch.randn(5, config.input_size)
    y_test_pred = model(X_test)
    print(f"Test input shape: {X_test.shape}")
    print(f"Test predictions shape: {y_test_pred.shape}")
    print(f"Sample predictions: {y_test_pred.squeeze().detach().numpy()}")
    print(f"True test output: {X_test @ true_weights + true_bias}")
    print("=== End of training loop ===")


if __name__ == "__main__":
    run()
