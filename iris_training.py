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
import sklearn


@dataclasses.dataclass(frozen=True)
class IrisTrainingConfig:
    input_size: int = 3
    output_size: int = 1
    num_samples: int = 50
    learning_rate: float = 0.1
    seed: int = 42

    training_config: TrainingConfig = dataclasses.field(
        default_factory=lambda: TrainingConfig(
            num_epochs=50,
            eval_every_n_steps=10,
        )
    )


@dataclasses.dataclass
class DataItem:
    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class IrisModel(nn.Module):
    """Simple Iris model: y = Wx + b"""

    def __init__(self, input_size, hidden_size, output_size):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        print(f"Some parameters of the iris model: {self.fc1.weight.data[0:5, 0:5]}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class IrisTrainingState(TrainingState[DataItem]):
    """Training state for Iris model"""

    def __init__(self, model: IrisModel, config: IrisTrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # linear learning rate schedule with warmup
        switch_step = config.training_config.num_epochs * 0.05
        linear_warmup = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=switch_step
        )
        linear_decay = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.training_config.num_epochs - switch_step,
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[linear_warmup, linear_decay],
            milestones=[switch_step],
        )

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def print_parameters(self) -> str:
        return f"{self.model.fc1.weight.data} {self.model.fc1.bias.data}"

    def step(self, data: DataItem) -> Metrics:
        # X, y, true_weights, true_bias = data
        # Forward pass
        predictions = self.model(data.inputs)
        loss = self.criterion(predictions, data.targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")

        # detach loss
        loss_numpy = float(loss.detach().numpy())

        return {
            "loss": loss_numpy,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_norm": float(self.model.fc1.weight.data.norm()),
            "bias_norm": float(self.model.fc1.bias.data.norm()),
            "weight_std": float(self.model.fc1.weight.data.std()),
            "bias_std": float(self.model.fc1.bias.data.std()),
            "weight_min": float(self.model.fc1.weight.data.min()),
            "bias_min": float(self.model.fc1.bias.data.min()),
            "weight_max": float(self.model.fc1.weight.data.max()),
            "bias_max": float(self.model.fc1.bias.data.max()),
            "weight_mean": float(self.model.fc1.weight.data.mean()),
            "bias_mean": float(self.model.fc1.bias.data.mean()),
        }

    def eval(self, data: DataItem) -> Metrics:
        predictions = self.model(data.inputs)
        loss = self.criterion(predictions, data.targets)
        return {"loss": float(loss.detach().numpy())}


class IrisDataGenerator(DataProvider[DataItem]):
    """Data generator for Iris data"""

    def __init__(self, config: TrainingConfig, is_train: bool = True):
        self.config = config
        iris = sklearn.datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2, random_state=config.seed
        )
        self.inputs_train = torch.tensor(X_train, dtype=torch.float32)
        self.targets_train = torch.tensor(y_train, dtype=torch.long)
        self.inputs_test = torch.tensor(X_test, dtype=torch.float32)
        self.targets_test = torch.tensor(y_test, dtype=torch.long)
        self.is_train = is_train

    def generate(self) -> Iterable[DataItem]:
        """Generate Iris data"""
        if self.is_train:
            yield DataItem(self.inputs_train, self.targets_train, metadata={})
        else:
            yield DataItem(self.inputs_test, self.targets_test, metadata={})

    def get_name(self) -> str:
        """Name of the dataset"""
        return f"Iris dataset ({self.config.seed=}, {self.is_train=})"


def train_iris_model(config: TrainingConfig):
    """Train an Iris model using configuration object"""
    # Create model
    with torch.random.fork_rng():
        torch.random.manual_seed(config.seed)
        model = IrisModel(config.input_size, 16, config.output_size)
    # Create training state
    state = IrisTrainingState(model, config)
    # Create data generator
    data_provider = IrisDataGenerator(config)
    eval_data_provider = IrisDataGenerator(config, is_train=False)
    # Train the model
    losses = train(
        state,
        data_provider,
        config.training_config,
        eval_data_providers=(eval_data_provider,),
    )
    return losses


if __name__ == "__main__":
    config = IrisTrainingConfig(
        learning_rate=0.2,
        input_size=4,
        output_size=3,
        num_samples=10,
        seed=42,
        training_config=TrainingConfig(
            num_epochs=50,
            eval_every_n_steps=10,
        ),
    )
    losses = train_iris_model(config)
