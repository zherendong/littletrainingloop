import random

from aggregation import (
    _quantize_relative,
    ExactMetricsAggregator,
    ApproximateMetricsAggregator,
)
import prng


def test_quantize_relative_zero():
    assert _quantize_relative(0.0, 0.05) == 0.0


def _generate_test_numbers():
    test_numbers = []
    with prng.PRNG(42333):
        test_numbers.extend(random.sample(range(1, 1000000), 100))
        for _ in range(100):
            test_numbers.append(random.uniform(-0.1, 2))
        return test_numbers


def test_quantize_relative_is_close():
    test_numbers = _generate_test_numbers()
    precision_settings = [0.01, 0.05, 0.1, 0.2, 0.5]
    for test_number in test_numbers:
        for precision in precision_settings:
            assert abs(_quantize_relative(test_number, precision) - test_number) < abs(
                test_number * precision
            )


def test_quantize_relative_maps_close_numbers_to_same_value():
    test_numbers = _generate_test_numbers()
    precision_settings = [0.01, 0.05, 0.1, 0.2, 0.5]
    for test_number in test_numbers:
        if test_number == 0.0:
            continue
        for precision in precision_settings:
            deviation = (
                test_number * precision / 3
            )  # 2 would be sufficient to guarantee precision, but in practice we seem to use a few more buckets than needed
            up = test_number + deviation
            down = test_number - deviation
            up_is_same = _quantize_relative(up, precision) == _quantize_relative(
                test_number, precision
            )
            down_is_same = _quantize_relative(down, precision) == _quantize_relative(
                test_number, precision
            )
            assert (
                up_is_same or down_is_same
            ), f"{up=}, {down=}, {test_number=}, {precision=}"


def test_exact_metrics_aggregator():
    exact = ExactMetricsAggregator()
    test_numbers = list(range(100))
    random.shuffle(test_numbers)
    for test_number in test_numbers:
        exact.observe(test_number)
    assert exact.mean() == sum(test_numbers) / len(test_numbers)
    assert exact.min() == min(test_numbers)
    assert exact.max() == max(test_numbers)
    assert exact.count() == len(test_numbers)
    assert exact.sum() == sum(test_numbers)
    assert exact.quantile(0.5) == 49
    assert exact.quantile(0.9) == 89
    assert exact.quantile(0.1) == 9
    assert exact.quantile(0.0) == 0
    assert exact.quantile(1.0) == 99
    assert exact.std() == 28.86607004772212


def test_approximate_metrics_aggregator():
    precision = 0.05
    approx = ApproximateMetricsAggregator(precision=precision)
    test_numbers = list(range(100))
    random.shuffle(test_numbers)
    for test_number in test_numbers:
        approx.observe(test_number)
    assert approx.mean() == sum(test_numbers) / len(test_numbers)
    assert approx.min() == min(test_numbers)
    assert approx.max() == max(test_numbers)
    assert approx.count() == len(test_numbers)
    assert approx.sum() == sum(test_numbers)
    assert abs(approx.quantile(0.5) - 49) < 49 * precision
    assert abs(approx.quantile(0.9) - 89) < 89 * precision
    assert abs(approx.quantile(0.1) - 9) < 9 * precision
    assert abs(approx.quantile(0.11) - 10) < 10 * precision
    assert abs(approx.quantile(0.12) - 11) < 11 * precision
    assert abs(approx.quantile(0.0)) == 0
    assert abs(approx.quantile(1.0) - 99) < 99 * precision
    assert abs(approx.std() - 28.86607004772212) < 29 * precision
    assert approx.get_num_buckets() < 60
