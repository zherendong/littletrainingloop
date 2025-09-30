"""Null implementation for neptune API.

Meant to allow running experiments without reporting and simplify the code that reports to neptune.
"""


class NullNeptuneFloatSeries:

    def fetch_value(self):
        raise NotImplementedError()

    def append(self, value, step=None):
        pass


class NullNeptuneRun:

    def __getitem__(self, key):
        return NullNeptuneRun()

    def __setitem__(self, key, value):
        pass

    def stop(self):
        pass
