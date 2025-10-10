"""Entry point for running a series of LLM training experiments.

- Manages a single scaling experiment with multiple LLM training runs
  - at different scale
  - at same scale with different hyper parameters, like
    - learning rate, because stability
    - x * Chinchilla data scaling, because data efficiency
    These hyper parameters are intended to be "within" the scale,
    i.e. we want to take the best hyper parameter combination for each scale.
- Determine compute pareto frontier for a scaling experiment.
- Manage multiple GPUs
  - scheduling multiple runs in parallel
  - multi-GPU training runs
- Notify per email when an experiment series or single run is done or failed.

"""

import argparse
import os
import subprocess
import time
import datetime

class TargetState: