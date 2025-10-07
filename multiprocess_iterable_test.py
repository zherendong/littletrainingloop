import multiprocess_iterable
import dataclasses
from typing import TypeVar, Iterable, Iterator
import time

T = TypeVar("T")


@dataclasses.dataclass
class Batch:
    data: list


class MyBatcher(Iterable[T]):
    def __init__(self, n, batch_size=4, delay=0.0):
        self.n = n
        self.batch_size = batch_size
        self.delay = delay  # simulate work

    def __iter__(self) -> Iterator[T]:
        buf = []
        for i in range(self.n):
            buf.append(i)
            if len(buf) == self.batch_size:
                if self.delay:
                    time.sleep(self.delay)
                yield Batch(buf)
                buf = []
        if buf:
            if self.delay:
                time.sleep(self.delay)
            yield Batch(buf)


def test_multiprocess_iterable():
    it = multiprocess_iterable.GeneratorProcess(
        MyBatcher,
        {"n": 17, "batch_size": 5, "delay": 0.01},
        prefetch=4,
    )
    try:
        for idx, batch in enumerate(it):
            print(batch)  # Batch(data=[...])
            # break early to test clean shutdown
            if idx >= 2:
                break
    finally:
        it.close()
    assert False
