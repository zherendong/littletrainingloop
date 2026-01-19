"""
Learning rate scheduler with per-block warmup support.

When a new block is added, it gets a fresh warmup that ramps up independently
of the global schedule.
"""

import torch.optim as optim
from typing import Callable


class BlockAwareScheduler:
    """Wraps a base scheduler and applies per-block warmup multipliers.
    
    Usage:
        optimizer = AdamW([{'params': model.parameters()}], lr=base_lr)
        base_scheduler = LinearLR(optimizer, ...)
        scheduler = BlockAwareScheduler(optimizer, base_scheduler, warmup_steps=1000)
        
        # Later, when adding a new block:
        new_params = model.add_block()
        scheduler.add_param_group(new_params, current_step=5000)
        
        # Each step:
        scheduler.step()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_scheduler: optim.lr_scheduler.LRScheduler,
        warmup_steps: int = 1000,
        warmup_start_factor: float = 0.01,
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.current_step = 0

        # Track when each param group was added
        # group_idx -> step when added (0 for initial groups)
        self.group_start_steps: dict[int, int] = {}
        for i in range(len(optimizer.param_groups)):
            self.group_start_steps[i] = 0

    def add_param_group(self, params, current_step: int | None = None):
        """Add a new parameter group with fresh warmup."""
        step = current_step if current_step is not None else self.current_step
        
        # Get current base LR from first group
        base_lr = self.optimizer.param_groups[0]["lr"]
        
        # Add to optimizer with starting LR (will be adjusted by warmup)
        param_group = {"params": list(params), "lr": base_lr * self.warmup_start_factor}
        self.optimizer.add_param_group(param_group)
        
        # Track start step
        new_idx = len(self.optimizer.param_groups) - 1
        self.group_start_steps[new_idx] = step

    def _compute_warmup_factor(self, group_idx: int) -> float:
        """Compute warmup multiplier for a param group."""
        start_step = self.group_start_steps.get(group_idx, 0)
        steps_since_start = self.current_step - start_step
        
        if steps_since_start >= self.warmup_steps:
            return 1.0
        
        # Linear warmup from warmup_start_factor to 1.0
        progress = steps_since_start / self.warmup_steps
        return self.warmup_start_factor + progress * (1.0 - self.warmup_start_factor)

    def step(self):
        """Take a scheduler step, applying per-group warmup."""
        self.current_step += 1
        
        # First, let base scheduler update (affects all groups uniformly)
        self.base_scheduler.step()
        
        # Get the base LR (from group 0, which has full warmup)
        base_lr = self.optimizer.param_groups[0]["lr"]
        
        # Apply warmup factors to newer groups
        for i in range(1, len(self.optimizer.param_groups)):
            warmup_factor = self._compute_warmup_factor(i)
            self.optimizer.param_groups[i]["lr"] = base_lr * warmup_factor

    def get_last_lr(self) -> list[float]:
        """Return last LR for each param group."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "current_step": self.current_step,
            "group_start_steps": self.group_start_steps,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self.current_step = state_dict["current_step"]
        self.group_start_steps = state_dict["group_start_steps"]


def create_block_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    global_warmup_steps: int,
    block_warmup_steps: int = 1000,
) -> BlockAwareScheduler:
    """Create a scheduler with linear warmup + linear decay, plus per-block warmup.
    
    Args:
        optimizer: The optimizer to schedule
        total_steps: Total training steps
        global_warmup_steps: Warmup steps for the global schedule
        block_warmup_steps: Warmup steps for each new block
    """
    # Global schedule: linear warmup then linear decay
    linear_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=global_warmup_steps,
    )
    linear_decay = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps - global_warmup_steps,
    )
    base_scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_warmup, linear_decay],
        milestones=[global_warmup_steps],
    )
    
    return BlockAwareScheduler(
        optimizer,
        base_scheduler,
        warmup_steps=block_warmup_steps,
    )

