import dataclasses
from typing import Iterable, Any

from training_basics import (
    TrainingConfig,
    EvalConfig,
    DataProvider,
    TrainingState,
    Metrics,
    MetricItem,
)
from training_loop import train
import neptune_lib
import torch
import torch.nn as nn
import torch.optim as optim


@dataclasses.dataclass(frozen=True)
class LinearTrainingConfig:
    """Configuration for linear model training"""

    input_size: int = 10
    output_size: int = 1
    num_samples: int = 100
    learning_rate: float = 0.01
    seed: int = 42
    training_config: TrainingConfig = dataclasses.field(
        default_factory=lambda: TrainingConfig(num_epochs=10)
    )
    eval_config: EvalConfig = dataclasses.field(
        default_factory=lambda: EvalConfig(every_n_steps=100)
    )


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

    def __init__(self, model: LinearModel, config: LinearTrainingConfig):
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
        loss_numpy = float(loss.detach().cpu().numpy())

        return {
            "loss": MetricItem(loss_numpy),
        }

    def validation_loss(
        self, eval_data: Iterable[DataItem], eval_steps: int
    ) -> Metrics:
        """Compute validation loss on the entire dataset."""
        raise NotImplementedError

    def evaluate(self) -> Metrics:
        raise NotImplementedError

    def save_checkpoint(self, path: str, run_id: str, step: int, epoch: int) -> None:
        pass

    def num_non_embedding_parameters(self) -> int:
        return self.num_parameters()


def generate_random_data(
    config: LinearTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random input-output pairs using configuration"""
    with torch.random.fork_rng():
        torch.random.manual_seed(config.seed)
        X = torch.randn(config.num_samples, config.input_size)
        y = torch.randn(config.num_samples, config.output_size)
        true_weights = torch.randn(config.input_size, config.output_size)
        true_bias = torch.randn(config.output_size)
        return X, y, true_weights, true_bias


class RandomLinearDataGenerator(DataProvider[DataItem]):
    """Data generator for random linear data"""

    def __init__(self, config: LinearTrainingConfig):
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


def train_linear_model(config: LinearTrainingConfig):
    """Train a linear model using configuration object"""
    # Create model
    model = LinearModel(config.input_size, config.output_size)
    # Create training state
    state = LinearModelTrainingState(model, config)
    # Create data generator
    data_provider = RandomLinearDataGenerator(config)
    # Train the model
    train(
        state,
        data_provider,
        config=config.training_config,
        eval_config=config.eval_config,
        neptune_run=neptune_lib.NullNeptuneRun(),
    )
    return
