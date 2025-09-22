"""
Abstract classes and configuration classes for the training loop.
"""

import abc
import dataclasses
from typing import Generic, Iterable, TypeVar


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration class for training hyperparameters"""

    num_epochs: int = 1
    training_steps_per_epoch: int = 100

    eval_every_n_steps: int = 20
    eval_steps: int = 1
    eval_batch_size: int = 32


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
