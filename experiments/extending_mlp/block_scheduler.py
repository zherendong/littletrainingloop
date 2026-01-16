"""
Learning rate scheduler with per-block warmup support.

Each block follows the same schedule shape (warmup + decay), but offset by
when it was added. A block added at step 500 sees "step 0" of its schedule.
"""

import abc
import torch.optim as optim


class LearningRateSchedule(abc.ABC):

    @abc.abstractmethod
    def get_lr_for_step(self, step: int) -> float:
        """Get LR for a given step."""
        pass


class LinearLRSchedule(LearningRateSchedule):
    def __init__(
        self,
        num_steps: int,
        start_lr: float,
        end_lr: float,
        hold_after_completion: bool = False,
    ):
        self.num_steps = num_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.hold_after_completion = hold_after_completion

    def get_lr_for_step(self, step: int) -> float:
        """Get LR for a given step."""
        if step > self.num_steps and self.hold_after_completion:
            return self.end_lr
        if step < 0 or step > self.num_steps:
            raise ValueError(f"Step {step} out of range (0, {self.num_steps})")
        progress = step / self.num_steps
        return self.start_lr + progress * (self.end_lr - self.start_lr)


class LinearWarmupLinearDecay(LearningRateSchedule):
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int,
        min_lr_factor: float = 0.1,
        hold_after_completion: bool = True,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_factor = min_lr_factor

        self.warmup_schedule = LinearLRSchedule(
            warmup_steps, start_lr=min_lr_factor, end_lr=1.0
        )
        self.decay_schedule = LinearLRSchedule(
            total_steps - warmup_steps,
            start_lr=1.0,
            end_lr=min_lr_factor,
            hold_after_completion=hold_after_completion,
        )

    def get_lr_for_step(self, step: int) -> float:
        """Get LR for a given step."""
        if step < self.warmup_steps:
            return self.warmup_schedule.get_lr_for_step(step)
        return self.decay_schedule.get_lr_for_step(step - self.warmup_steps)


class BlockAwareLRScheduler:
    """Scheduler where each param group follows the same schedule, offset by start time.

    Instead of wrapping a base scheduler and applying multipliers, we compute
    the LR directly for each group based on its effective step (current - start).

    Usage:
        optimizer = AdamW([{'params': model.parameters()}], lr=base_lr)
        scheduler = BlockAwareScheduler(optimizer, total_steps=10000, warmup_steps=1000)

        # Later, when adding a new block:
        new_params = model.add_block()
        scheduler.add_param_group(new_params)

        # Each step:
        scheduler.step()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        schedule: LearningRateSchedule,
    ):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.param_groups[0]["lr"])
        self.schedule = schedule
        self.current_step = 0

        # Track when each param group was added
        # group_idx -> step when added (0 for initial groups)
        self.group_start_steps: dict[int, int] = {}
        for i in range(len(optimizer.param_groups)):
            self.group_start_steps[i] = 0

    def add_param_group(self, params):
        """Add a new parameter group. It starts at step 0 of the schedule."""
        # Compute initial LR (step 0 of warmup)
        initial_lr = self.base_lr

        param_group = {"params": list(params), "lr": initial_lr}
        self.optimizer.add_param_group(param_group)

        # Track start step
        new_idx = len(self.optimizer.param_groups) - 1
        self.group_start_steps[new_idx] = self.current_step

    def step(self):
        """Take a scheduler step, updating LR for each group based on its offset."""
        self.current_step += 1

        for i, group in enumerate(self.optimizer.param_groups):
            start_step = self.group_start_steps[i]
            effective_step = self.current_step - start_step
            assert effective_step >= 0, f"Effective step {effective_step} < 0"
            group["lr"] = self.base_lr * self.schedule.get_lr_for_step(effective_step)

    def get_last_lr(self) -> list[float]:
        """Return last LR for each param group."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "group_start_steps": self.group_start_steps,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.group_start_steps = state_dict["group_start_steps"]
        self.base_lr = state_dict["base_lr"]


def create_block_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> BlockAwareLRScheduler:
    """Create a scheduler with linear warmup + linear decay, per-block offset.

    Args:
        optimizer: The optimizer to schedule
        total_steps: Total training steps (for decay calculation)
        warmup_steps: Warmup steps for each block's schedule
    """
    return BlockAwareLRScheduler(
        optimizer,
        schedule=LinearWarmupLinearDecay(total_steps, warmup_steps, min_lr_factor=0.0),
    )
