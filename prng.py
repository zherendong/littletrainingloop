import random
import torch

class PRNG:
    """Self-contained random number generator to make sure we don't pollute global state.

    Usage:

    prng = PRNG(42)
    print(prng.random())
    print(prng.random())

    Or:

    with prng:
        print(random.random())
        print(random.random())
    """

    def __init__(self, seed):
        self.original_state = random.getstate()
        self.seed = seed
        random.seed(self.seed)
        self.prng_state = random.getstate()
        random.setstate(self.original_state)

        self.torch_original_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        self.prng_torch_state = torch.random.get_rng_state()
        torch.random.set_rng_state(self.torch_original_state)

    def __enter__(self):
        self.original_state = random.getstate()
        random.setstate(self.prng_state)
        self.original_torch_state = torch.random.get_rng_state()
        torch.random.set_rng_state(self.prng_torch_state)

    def __exit__(self, exc_type, exc_value, traceback):
        self.prng_state = random.getstate()
        if self.original_state is not None:
            random.setstate(self.original_state)
        self.prng_torch_state = torch.random.get_rng_state()
        if self.original_torch_state is not None:
            torch.random.set_rng_state(self.original_torch_state)

    def random(self):
        with self:
            return random.random()

    def shuffle(self, x):
        with self:
            random.shuffle(x)

    def random_int(self, a, b):
        with self:
            return random.randint(a, b)
