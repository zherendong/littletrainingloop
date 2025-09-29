"""A generic metrics aggregator for approximate quantiles and other aggregate values."""

import abc
import math
from collections import defaultdict


class MetricsAggregator(abc.ABC):

    @abc.abstractmethod
    def observe(self, value: float) -> None:
        """Observe a value."""
        pass

    @abc.abstractmethod
    def quantile(self, q: float) -> float:
        """Return the q-th quantile"""
        pass

    @abc.abstractmethod
    def mean(self) -> float:
        """Return the mean"""
        pass

    @abc.abstractmethod
    def std(self) -> float:
        """Return the standard deviation"""
        pass

    @abc.abstractmethod
    def min(self) -> float:
        """Return the minimum"""
        pass

    @abc.abstractmethod
    def max(self) -> float:
        """Return the maximum"""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """Return the number of observations"""
        pass

    @abc.abstractmethod
    def sum(self) -> float:
        """Return the sum of all observations"""
        pass


class ExactMetricsAggregator(MetricsAggregator):
    """Exact metrics aggregator using a list to store all observations"""

    def __init__(self):
        self.values = []

    def observe(self, value: float, weight: int = 1) -> None:
        self.values.append(value)

    def quantile(self, q: float) -> float:
        if q < 0 or q > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}")
        index = math.floor(len(self.values) * q) - 1
        index = max(0, index)
        index = min(index, len(self.values) - 1)
        return sorted(self.values)[index]

    def mean(self) -> float:
        if len(self.values) == 0:
            print("Warning: mean of empty list")
            return 0.0
        return sum(self.values) / len(self.values)

    def std(self) -> float:
        return (
            sum((x - self.mean()) ** 2 for x in self.values) / len(self.values)
        ) ** 0.5

    def min(self) -> float:
        return min(self.values)

    def max(self) -> float:
        return max(self.values)

    def count(self) -> int:
        return len(self.values)

    def sum(self) -> float:
        return sum(self.values)


def _quantize_relative(value: float, precision: float) -> float:
    """Quantize a value relative to its magnitude."""
    if value == 0.0:
        return 0.0
    sign = 1 if value > 0 else -1
    value = abs(value)
    base = 1 + precision
    return sign * (base ** round(math.log(value) / math.log(base)))


class ApproximateMetricsAggregator(MetricsAggregator):
    """Approximate metrics aggregator using a histogram to store observations"""

    def __init__(self, precision: float = 0.05):
        self._precision = precision

        self._min_value = float("inf")
        self._max_value = float("-inf")
        self._sum = 0.0
        self._count = 0

        self.buckets = defaultdict(int)

    def observe(self, value: float) -> None:
        self._min_value = min(self._min_value, value)
        self._max_value = max(self._max_value, value)
        self._sum += value
        self._count += 1

        bucket_key = _quantize_relative(value, self._precision)
        self.buckets[bucket_key] += 1

    def quantile(self, q: float) -> float:
        if q < 0 or q > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}")
        target_count = self._count * q
        # print(f"{target_count=}")
        current_count = 0
        for bucket_key, bucket_count in sorted(
            self.buckets.items(), key=lambda x: x[0]
        ):
            current_count += bucket_count
            # print(f"{current_count=}, {bucket_key=:.3f}")
            if current_count >= target_count:
                return bucket_key
        return self._max_value

    def mean(self) -> float:
        return self._sum / self._count

    def std(self) -> float:
        mean = self.mean()
        squared_diff_sum = 0.0
        for bucket_key, bucket_count in self.buckets.items():
            squared_diff_sum += bucket_count * (bucket_key - mean) ** 2
        return (squared_diff_sum / self._count) ** 0.5

    def min(self) -> float:
        return self._min_value

    def max(self) -> float:
        return self._max_value

    def count(self) -> int:
        return self._count

    def sum(self) -> float:
        return self._sum

    def get_num_buckets(self) -> int:
        return len(self.buckets)
