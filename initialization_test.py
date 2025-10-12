import torch
import pytest
import math

from transformer import CopyLinear


@pytest.mark.parametrize("shape", [(128, 512), (512, 128), (512, 512), (1024, 4096)])
def test_copy_linear_init_std_of_inputs_is_approx_maintained(shape):
    input_size, output_size = shape
    model = CopyLinear(input_size, output_size, dtype=torch.float32)
    print(f"{torch.std(model.linear.weight)=}")
    print(f"expected std: {1.0 / math.sqrt(input_size)}")

    x = torch.normal(
        mean=0,
        std=1,
        size=(128, input_size),
        dtype=torch.float32,
    )
    x_std = torch.std(x)
    print(f"{x_std=}")
    y = model(x)
    y_std = torch.std(y)
    print(f"{y_std=}")
    assert y_std < x_std
    assert y_std * 2 > x_std


def test_diag():
    a = 16
    b = 8

    x = torch.arange(a, dtype=torch.float32).unsqueeze(0)
    y = torch.arange(b, dtype=torch.float32).unsqueeze(1)

    # compress to range from 0 to 1
    x = x / (a - 1)
    y = y / (b - 1)

    print(f"{x=}")
    print(f"{y=}")
    print(f"{torch.abs(x - y)=}")

    diff = 1 + 10 * torch.abs(x - y)
    diff_diag = 1 / (diff * diff)
    print(f"{diff=}")
    print(f"{diff_diag=}")
