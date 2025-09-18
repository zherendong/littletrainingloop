"""
Generic training loop to understand the minimal requirements for a training loop.

Design goals:
- no torch, numpy, etc
- generate a clean list of functions to implement
"""

import abc
from typing import Generic, Iterable, TypeVar

import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration class for training hyperparameters"""

    input_size: int = 3
    output_size: int = 1
    num_samples: int = 50
    num_epochs: int = 100
    learning_rate: float = 0.1


Metrics = dict[str, float]


D = TypeVar("D")


class DataProvider(Generic[D], abc.ABC):
    """Abstract base class for data generation"""

    @abc.abstractmethod
    def generate(self) -> Iterable[D]:
        """Generate data using configuration.

        Returns batches of data, ends at the end of the dataset (=epoch).
        """
        pass


class TrainingState(Generic[D], abc.ABC):
    """Abstract base class for training state"""

    @abc.abstractmethod
    def num_parameters(self):
        """Number of parameters in the model"""
        pass

    @abc.abstractmethod
    def print_parameters(self) -> str:
        """Print model parameters"""
        pass

    @abc.abstractmethod
    def step(self, data: D) -> Metrics:
        """Take a training step, return metrics."""
        pass


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
):
    """Training loop using configuration object"""
    random_seed = 42
    torch.manual_seed(random_seed)

    losses = []
    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        if epoch == 0:
            print(f"Number of parameters: {state.num_parameters()}")
            # print(f"Initial parameters: {state.print_parameters()}")

        for idx, data in enumerate(data_provider.generate()):
            metrics = state.step(data)
            loss = metrics["loss"]
            losses.append(loss)
            if idx > 0 and idx % 10 == 0:
                print(f"Batch [{idx}], Loss: {loss:.4f}")

            # print metrics sorted by name
            # print(f"Batch [{idx}], metrics:")
            # for name, value in sorted(metrics.items()):
            #     print(f"  {name}: {value:.4f}")

        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss:.4f}")

    print("-" * 50)
    # print(f"Final parameters: {state.print_parameters()}\n")
    print("Training completed!")
    return losses
