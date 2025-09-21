"""
Test script for the prng.py module.
"""

import random
from prng import PRNG


def test_prng_different_seeds():
    """Different seeds should give different results."""
    prng1 = PRNG(1337)
    prng2 = PRNG(1338)
    assert prng1.random() != prng2.random()


def test_prng_state_progression():
    """Each call to random() should give a different result."""
    prng = PRNG(1337)
    assert prng.random() != prng.random()


def test_prng_results_not_affected_by_global_state():
    """Using the PRNG should not affect the global state."""
    prng1 = PRNG(1337)
    num1 = prng1.random()
    num2 = prng1.random()

    prng2 = PRNG(1337)
    assert prng2.random() == num1
    random.random()  # this should not affect prng2
    assert prng2.random() == num2


def test_prng_determinism():
    """Same seed should give same results."""
    prng1 = PRNG(1337)
    prng2 = PRNG(1337)
    assert prng1.random() == prng2.random()


def test_prng_state_preservation():
    """Using the PRNG should not affect the global state."""
    random.random()
    state = random.getstate()
    prng = PRNG(1337)
    prng.random()
    assert random.getstate() == state


def test_prng_no_surprise_changes():
    """Hardcoded values for one seed to avoid surprise changes."""
    prng = PRNG(42)
    assert prng.random() == 0.6394267984578837
    assert prng.random() == 0.025010755222666936
    # in particular, they need to be different
