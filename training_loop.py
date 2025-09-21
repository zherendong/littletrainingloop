"""
Generic training loop to understand the minimal requirements for a training loop.

Design goals:
- no torch, numpy, etc
- generate a clean list of functions to implement
"""

import dataclasses
import abc
from typing import Generic, Iterable, TypeVar, Sequence


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration class for training hyperparameters"""

    num_epochs: int = 100
    eval_every_n_steps: int = 10


Metrics = dict[str, float]


D = TypeVar("D")


class DataProvider(Generic[D], abc.ABC):
    """Abstract base class for data generation"""

    @abc.abstractmethod
    def generate(self) -> Iterable[D]:
        """Generate data using configuration.

        Returns batches of data, ends at the end of the dataset (=epoch).
        Calling generate again starts from the beginning of the dataset.
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        """Name of the dataset"""
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

    @abc.abstractmethod
    def eval(self, data: D) -> Metrics:
        """Evaluate the model, return metrics."""
        pass


def print_metrics(metrics: Metrics) -> None:
    """Print metrics sorted by name"""
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")


def do_eval(
    state: TrainingState[D],
    eval_data_providers: Sequence[DataProvider[D]],
    epoch: int | None = None,
    step: int | None = None,
) -> Metrics:
    """Evaluate the model"""
    print(f"Eval metrics ({epoch=}, {step=}):")
    for eval_data_provider in eval_data_providers:
        print(f"  {eval_data_provider.get_name()}:")
        for data in eval_data_provider.generate():
            metrics = state.eval(data)
            print_metrics(metrics)


def train(
    state: TrainingState[D],
    data_provider: DataProvider[D],
    config: TrainingConfig,
    eval_data_providers: Sequence[DataProvider[D]] = (),
):
    """Training loop using configuration object"""

    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        for idx, data in enumerate(data_provider.generate()):
            if idx % config.eval_every_n_steps == 0:
                do_eval(state, eval_data_providers, epoch, idx)
            metrics = state.step(data)
            print(f"Step in epoch {idx}:")
            print_metrics(metrics)

        print(f"Epoch {epoch + 1} completed.")
        do_eval(state, eval_data_providers, epoch)

    print("-" * 50)
    print("Training completed!")
    return
