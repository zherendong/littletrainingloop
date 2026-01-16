"""
Learning rate scheduler with per-block warmup support.

Each block follows the same schedule shape (warmup + decay), but offset by
when it was added. A block added at step 500 sees "step 0" of its schedule.
"""

import math

import block_scheduler


def test_linear_schedule_start():
    """Test that linear schedule starts at start_lr."""
    schedule = block_scheduler.LinearLRSchedule(100, start_lr=0.1, end_lr=0.2)
    assert schedule.get_lr_for_step(0) == 0.1, "Should start at start_lr"


def test_linear_schedule_end():
    """Test that linear schedule ends at end_lr."""
    schedule = block_scheduler.LinearLRSchedule(100, start_lr=0.1, end_lr=0.2)
    assert schedule.get_lr_for_step(100) == 0.2, "Should end at end_lr"


def test_linear_schedule_mid():
    """Test that linear schedule is correct at mid-point."""
    schedule = block_scheduler.LinearLRSchedule(100, start_lr=0.1, end_lr=0.2)
    assert math.isclose(
        schedule.get_lr_for_step(50), 0.15
    ), "Should be halfway at step 50"


def test_linear_schedule_hold():
    """Test that linear schedule holds after completion if specified."""
    schedule = block_scheduler.LinearLRSchedule(
        100, start_lr=0.1, end_lr=0.2, hold_after_completion=True
    )
    assert schedule.get_lr_for_step(101) == 0.2, "Should hold after completion"
    assert schedule.get_lr_for_step(200) == 0.2, "Should hold after completion"
