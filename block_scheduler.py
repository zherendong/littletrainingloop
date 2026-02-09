"""
Learning rate scheduler with per-block warmup support for growing models.
"""

import abc
from typing import Iterator
import torch
import torch.optim as optim


class LearningRateSchedule(abc.ABC):

    @abc.abstractmethod
    def get_lr_for_step(self, step: int) -> float:
        """Get LR multiplier for a given step."""
        pass


class LinearWarmupLinearDecay(LearningRateSchedule):
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int,
        min_lr_factor: float = 0.1,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_factor = min_lr_factor

    def get_lr_for_step(self, step: int) -> float:
        """Get LR multiplier for a given step."""
        if step < 0:
            raise ValueError(f"Step {step} < 0")
        
        if step >= self.total_steps:
            return self.min_lr_factor
        
        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            return self.min_lr_factor + progress * (1.0 - self.min_lr_factor)
        
        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = (step - self.warmup_steps) / decay_steps
        return 1.0 - decay_progress * (1.0 - self.min_lr_factor)


class GrowingModelScheduler:
    """Scheduler supporting growing models with per-block LR schedules.
    
    Each parameter group can have its own schedule length. When a block is added,
    you specify how many steps its schedule should run for.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        min_lr_factor: float = 0.1,
        initial_schedule_steps: int | None = None,
    ):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.param_groups[0]["lr"])
        self.warmup_steps = warmup_steps
        self.min_lr_factor = min_lr_factor
        self.current_step = 0

        self.group_info: dict[int, tuple[int, LearningRateSchedule]] = {}
        
        for i in range(len(optimizer.param_groups)):
            if initial_schedule_steps is None:
                raise ValueError("Must specify initial_schedule_steps")
            schedule = LinearWarmupLinearDecay(
                total_steps=initial_schedule_steps,
                warmup_steps=warmup_steps,
                min_lr_factor=min_lr_factor,
            )
            self.group_info[i] = (0, schedule)

    def add_param_group(
        self,
        params: Iterator[torch.nn.Parameter],
        schedule_steps: int,
    ):
        """Add a new parameter group with its own schedule length."""
        initial_lr = self.base_lr * self.min_lr_factor
        # initial_lr = 0.0001
        
        param_group = {"params": list(params), "lr": initial_lr}
        self.optimizer.add_param_group(param_group)

        new_idx = len(self.optimizer.param_groups) - 1
        schedule = LinearWarmupLinearDecay(
            total_steps=schedule_steps,
            warmup_steps=self.warmup_steps,
            min_lr_factor=self.min_lr_factor,
        )
        self.group_info[new_idx] = (self.current_step, schedule)

    def step(self):
        """Take a scheduler step, updating LR for each group."""
        self.current_step += 1

        for i, group in enumerate(self.optimizer.param_groups):
            start_step, schedule = self.group_info[i]
            effective_step = self.current_step - start_step
            group["lr"] = self.base_lr * schedule.get_lr_for_step(effective_step)

    def get_last_lr(self) -> list[float]:
        """Return last LR for each param group."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "group_info": {
                i: (start, {
                    "total_steps": sched.total_steps,
                    "warmup_steps": sched.warmup_steps,
                    "min_lr_factor": sched.min_lr_factor,
                })
                for i, (start, sched) in self.group_info.items()
            },
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "min_lr_factor": self.min_lr_factor,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.base_lr = state_dict["base_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.min_lr_factor = state_dict["min_lr_factor"]
        self.group_info = {}
        for i, (start, sched_dict) in state_dict["group_info"].items():
            schedule = LinearWarmupLinearDecay(
                total_steps=sched_dict["total_steps"],
                warmup_steps=sched_dict["warmup_steps"],
                min_lr_factor=sched_dict["min_lr_factor"],
            )
            self.group_info[int(i)] = (start, schedule)