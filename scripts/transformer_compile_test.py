"""Experiments with torch.compile"""

import numpy as np
import time
import torch
import transformer
import model_configs.chinchilla  # noqa: F401


if torch.cuda.is_available():
    torch.set_default_device("cuda")


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


class Trainer:
    def __init__(self, model, opt=False):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters())
        self.crit = torch.nn.CrossEntropyLoss()

        if opt:
            # self.model_opt = torch.compile(self.model, mode="reduce-overhead")
            self.train = torch.compile(self.train, mode="reduce-overhead")
        else:
            # self.model_opt = self.model
            pass

    def train(self, data, targets):
        pred = self.model(data)
        pred = pred.view(-1, pred.shape[-1])
        targets = targets.view(-1)
        loss = self.crit(pred, targets)
        print(f"in the middle {loss.item()=}")
        self.opt.zero_grad(True)
        loss.backward()
        self.opt.step()


def run():
    config = transformer.transformer_config_registry.get("chinchilla-44m")
    vocab_size = 128
    model = transformer.TransformerModel(vocab_size=vocab_size, config=config)

    trainer = Trainer(model)
    trainer_opt = Trainer(model, opt=True)

    batch_size = 64
    sequence_length = 512

    eager_times = []
    for _ in range(10):
        inputs = torch.randint(
            0, vocab_size, (batch_size, sequence_length), dtype=torch.long
        )
        targets = torch.randint(
            0, vocab_size, (batch_size, sequence_length), dtype=torch.long
        )
        start = time.time()
        trainer.train(inputs, targets)
        torch.cuda.synchronize()
        end = time.time()
        eager_times.append(end - start)
        print(f"eager time: {end - start}")

    compile_times = []
    # model_opt = torch.compile(model, mode="reduce-overhead")
    for _ in range(10):
        inputs = torch.randint(
            0, vocab_size, (batch_size, sequence_length), dtype=torch.long
        )
        targets = torch.randint(
            0, vocab_size, (batch_size, sequence_length), dtype=torch.long
        )
        start = time.time()
        trainer_opt.train(inputs, targets)
        torch.cuda.synchronize()
        end = time.time()
        compile_times.append(end - start)
        print(f"compile time: {end - start}")

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    print(
        f"eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
    )


if __name__ == "__main__":
    run()
