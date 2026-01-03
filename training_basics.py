"""
Abstract classes and configuration classes for the training loop.
"""

import abc
import dataclasses
from typing import Generic, Iterable, TypeVar

import prng


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration class for training hyperparameters"""

    num_epochs: int = 1
    training_steps_per_epoch: int | None = 100
    seed: int = 42
    train_metrics_every_n_steps: int = 100
    checkpoint_path: str | None = None
    save_checkpoint_every_n_steps: int | None = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Configuration class for evaluation"""

    every_n_steps: int = 100
    steps: int = 1  # maximum number of batches to evaluate
    full_eval_every_n_steps: int | None = None


@dataclasses.dataclass(frozen=True)
class MetricItem:
    value: float
    x_axis: float | None = None  # if None, use step number


Metrics = dict[str, MetricItem]


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


class ShuffleBuffer(DataProvider[D]):
    """Shuffle data using a buffer."""

    def __init__(
        self,
        config: TrainingConfig,
        buffer_size: int,
        data_provider: DataProvider[D],
        name: str | None = None,
    ):
        self.config = config
        self.data_provider = data_provider
        assert buffer_size > 0, "Buffer size must be positive"
        self.buffer_size = buffer_size
        self.prng = prng.PRNG(config.seed + 6473284)
        self.name = name

    def generate(self) -> Iterable[D]:
        """Generate data using configuration.

        Returns batches of data, ends at the end of the dataset (=epoch).
        Calling generate again starts from the beginning of the dataset.
        """

        buffer = []
        for data in self.data_provider.generate():
            if len(buffer) < self.buffer_size:
                buffer.append(data)
            else:
                idx = self.prng.random_int(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = data
        self.prng.shuffle(buffer)
        for data in buffer:
            yield data

    def get_name(self) -> str:
        """Name of the dataset"""
        if self.name:
            return self.name
        return f"ShuffleBuffer({self.buffer_size}, {self.data_provider.get_name()})"


class TrainingState(Generic[D], abc.ABC):
    """Abstract base class for training state"""

    @abc.abstractmethod
    def num_non_embedding_parameters(self) -> int:
        """Number of parameters in the model, excluding embedding parameters"""
        pass

    @abc.abstractmethod
    def num_parameters(self) -> int:
        """Number of parameters in the model"""
        pass

    @abc.abstractmethod
    def step(self, data: D) -> Metrics:
        """Take a training step, return metrics."""
        pass

    @abc.abstractmethod
    def evaluate(self) -> Metrics:
        """Run the evaluation harness on the model."""
        pass

    @abc.abstractmethod
    def validation_loss(self, eval_data: Iterable[D], eval_steps: int) -> Metrics:
        """Compute validation loss on the entire dataset."""
        pass

    @abc.abstractmethod
    def save_checkpoint(self, path: str, run_id: str, step: int, epoch: int):
        """Save a training checkpoint."""
        pass
