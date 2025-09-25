import dataclasses
from typing import Iterable, Any

from training_loop import (
    TrainingConfig,
    DataProvider,
    TrainingState,
    Metrics,
    train,
)
import torch
import torch.nn as nn
import torch.optim as optim


@dataclasses.dataclass
class DataItem:
    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class LinearModel(nn.Module):
    """Simple linear model: y = Wx + b"""

    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class LinearModelTrainingState(TrainingState[DataItem]):
    """Training state for linear model"""

    def __init__(self, model: LinearModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def step(self, data: DataItem) -> Metrics:
        # X, y, true_weights, true_bias = data
        # Forward pass
        predictions = self.model(data.inputs)
        loss = self.criterion(predictions, data.targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # detach loss
        loss_numpy = float(loss.detach().numpy())

        return {"loss": loss_numpy}


def generate_random_data(
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random input-output pairs using configuration"""
    with torch.random.fork_rng():
        torch.random.manual_seed(42)
        X = torch.randn(config.num_samples, config.input_size)
        y = torch.randn(config.num_samples, config.output_size)
        true_weights = torch.randn(config.input_size, config.output_size)
        true_bias = torch.randn(config.output_size)
        return X, y, true_weights, true_bias


class RandomLinearDataGenerator(DataProvider[DataItem]):
    """Data generator for random linear data"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.X, self.y, self.true_weights, self.true_bias = generate_random_data(config)

    def generate(self) -> Iterable[DataItem]:
        """Generate random linear data"""
        yield DataItem(
            self.X,
            self.y,
            metadata={"true_weights": self.true_weights, "true_bias": self.true_bias},
        )

    def get_name(self) -> str:
        """Name of the dataset"""
        return f"random_linear dataset with seed {self.config.seed}"


def train_linear_model(config: TrainingConfig):
    """Train a linear model using configuration object"""
    # Create model
    model = LinearModel(config.input_size, config.output_size)
    # Create training state
    state = LinearModelTrainingState(model, config)
    # Create data generator
    data_provider = RandomLinearDataGenerator(config)
    # Train the model
    losses = train(state, data_provider, config)
    return losses
