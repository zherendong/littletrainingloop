#!/usr/bin/env python3
"""
Run the training loop
"""

from training_loop import TrainingConfig
from linear_training import train_linear_model


def run():
    config = TrainingConfig(num_epochs=10, learning_rate=0.3, input_size=50, output_size=10, num_samples=10)
    losses = train_linear_model(config)
    print(f"Losses: {losses}")

if __name__ == "__main__":
    run()
