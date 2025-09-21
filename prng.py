import random


class PRNG:
    """Self-contained random number generator to make sure we don't pollute global state.

    Usage:

    prng = PRNG(42)
    print(prng.random())
    print(prng.random())
    """

    def __init__(self, seed):
        self.original_state = random.getstate()
        self.seed = seed
        random.seed(self.seed)
        self.state = random.getstate()
        random.setstate(self.original_state)

    def __enter__(self):
        self.original_state = random.getstate()
        random.setstate(self.state)

    def __exit__(self, exc_type, exc_value, traceback):
        self.state = random.getstate()
        if self.original_state is not None:
            random.setstate(self.original_state)

    def random(self):
        with self:
            return random.random()
