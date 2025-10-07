"""A way to run a generator in a separate process.

Usage:

# run MyBatcher(n=17, batch_size=5) in a separate process
it = GeneratorProcess(MyBatcher, {"n": 17, "batch_size": 5, "delay": 0.01}, prefetch=4)
for batch in it:
    print(batch)  # Batch(data=[...])
    # break early to test clean shutdown
    # break
"""

import multiprocessing
import traceback
from typing import TypeVar, Iterable, Iterator, Callable, Any


T = TypeVar("T")


_SENTINEL = object()  # end-of-stream marker
_ERR = object()  # error marker


def _worker_iterable(iterable: Iterable[T], q, stop_evt):
    """Run in child process: iterate and put() batches into q."""
    try:
        for batch in iterable:
            if stop_evt.is_set():
                break
            q.put(batch)  # batches must be picklable
        q.put(_SENTINEL)
    except Exception:
        # ship a compact traceback to the parent
        q.put((_ERR, traceback.format_exc()))


def _worker_iterable_factory(
    iterable_factory: Callable[[], Iterable[T]], iterable_factory_kwargs, q, stop_evt
):
    """
    Create iterable via factory function and then run it.

    Run in child process.
    """
    iterable = iterable_factory(**iterable_factory_kwargs)
    _worker_iterable(iterable, q, stop_evt)


class GeneratorProcess(Iterable[T]):
    """
    Wrap an iterable so iteration happens in a separate process,
    with bounded prefetch via Queue(maxsize).
    """

    def __init__(
        self,
        iterable_factory: Callable[..., Iterable[T]],
        iterable_factory_kwargs: dict[str, Any],
        prefetch: int,
        start_method="spawn",
    ):
        ctx = multiprocessing.get_context(start_method)
        self._ctx = ctx
        self._q = ctx.Queue(maxsize=prefetch)
        self._stop = ctx.Event()
        self._p = ctx.Process(  # type: ignore
            target=_worker_iterable_factory,
            args=(iterable_factory, iterable_factory_kwargs, self._q, self._stop),
            daemon=False,
        )
        self._p.start()
        self._done = False

    def __iter__(self) -> Iterator[T]:
        try:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    self._done = True
                    break
                if isinstance(item, tuple) and item and item[0] is _ERR:
                    # re-raise child exception with its traceback
                    _, tb = item
                    raise RuntimeError(f"Child process failed:\n{tb}")
                item: T = item
                yield item
        finally:
            self.close()

    def close(self):
        if not self._done:
            # parent stopped early -> tell child to exit
            self._stop.set()
        if self._p.is_alive():
            self._p.join(timeout=5)
            if self._p.is_alive():
                self._p.terminate()
        self._done = True
