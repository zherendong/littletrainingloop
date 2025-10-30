from typing import Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function

# default device cuda
device = "cuda"
torch.set_default_device(device)


# # from deep_gemm
# def _2d_per_channel_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Cast a 2D tensor to fp8 per channel."""
#     assert x.dim() == 2 and x.size(0) % 128 == 0
#     m, n = x.shape
#     x_view = x.view(-1, 128, n)
#     x_amax = x_view.abs().float().amax(dim=1).view(-1, n).clamp(1e-4)
#     scale = x_amax / 448.0
#     assert scale.shape == (m // 128, n)
#     return (x_view * (1.0 / scale.unsqueeze(1))).to(torch.float8_e4m3fn).view(
#         m, n
#     ), scale


# # from deep_gemm
# def per_channel_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Cast a tensor to fp8 per channel."""
#     assert x.ndim >= 2
#     orig_shape = x.shape
#     x = x.view(-1, x.shape[-1])
#     x, scale = _2d_per_channel_cast_to_fp8(x)
#     return x.view(orig_shape), scale


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, x, W):
        # x: (N, in_features), W: (in_features, out_features)
        y = F.linear(x, W)
        ctx.save_for_backward(x, W)
        assert y.is_contiguous()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # grad_y: (N, out_features)
        x, W = ctx.saved_tensors

        # dL/dx = grad_y @ W^T
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_y.matmul(W)
            # grad_x = F.linear(grad_y, W.T)
            assert grad_x.is_contiguous()

        # dL/dW = x^T @ grad_y
        grad_W = None
        if ctx.needs_input_grad[1]:
            # grad_W = F.linear(x.t(), grad_y.T)
            grad_W = grad_y.T.matmul(x)
            # grad_W = x.T.matmul(grad_y).T
            assert grad_W.is_contiguous()

        return grad_x, grad_W


class FP8Linear(Function):
    @staticmethod
    def forward(ctx, x, W):
        # x: (N, in_features), W: (in_features, out_features)
        x_fp8 = x.to(torch.float8_e4m3fn)
        W_fp8 = W.to(torch.float8_e4m3fn)
        scale_result = torch.empty(())
        scale_result.fill_(42.0)
        # https://github.com/pytorch/pytorch/blob/29516bd2a0fc1dbb437eed606dc41074e21f2b97/aten/src/ATen/native/cuda/Blas.cpp#L792
        y = torch._scaled_mm(
            input=x_fp8,
            mat2=W_fp8.T,
            bias=None,
            out_dtype=torch.float32,
            scale_a=torch.tensor(1.0),
            scale_b=torch.tensor(1.0),
            scale_result=scale_result,
        )
        print(f"{scale_result=}")
        ctx.save_for_backward(x, W)
        # print(f"{x.shape=}, {W_fp8.shape=}, {y.shape=}")
        assert y.dtype == torch.float32
        assert y.is_contiguous()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # grad_y: (N, out_features)
        x, W = ctx.saved_tensors

        # dL/dx = grad_y @ W^T
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_y.matmul(W)
            # grad_x = F.linear(grad_y, W.T)
            assert grad_x.is_contiguous()

        # dL/dW = x^T @ grad_y
        grad_W = None
        if ctx.needs_input_grad[1]:
            # grad_W = F.linear(x.t(), grad_y.T)
            grad_W = grad_y.T.matmul(x)
            # grad_W = x.T.matmul(grad_y).T
            assert grad_W.is_contiguous()

        return grad_x, grad_W


def compute_scale(x: torch.Tensor) -> torch.Tensor:
    x_max = x.abs().max()  # unnecessary copy via abs?
    scale = x_max / 448.0
    return scale


def to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast a tensor to fp8 and return the scale."""
    assert x.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert x.is_cuda
    with torch.no_grad():
        scale = compute_scale(x)
        # x_fp8 = x.to(torch.float8_e4m3fn)
        x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale


class LinearFullFP8(Function):
    @staticmethod
    def forward(ctx, x, W):

        # x: (N, in_features), W: (in_features, out_features)
        # x_fp8 = x.to(torch.float8_e4m3fn)
        # W_fp8 = W.to(torch.float8_e4m3fn)
        x_fp8, scale_x = to_fp8(x)
        W_fp8, scale_W = to_fp8(W)

        scale_result = torch.empty(())
        scale_result.fill_(42.0)
        # https://github.com/pytorch/pytorch/blob/29516bd2a0fc1dbb437eed606dc41074e21f2b97/aten/src/ATen/native/cuda/Blas.cpp#L792
        y = torch._scaled_mm(
            input=x_fp8,
            mat2=W_fp8.T,
            bias=None,
            out_dtype=torch.float32,
            scale_a=scale_x,
            scale_b=scale_W,
            scale_result=scale_result,
            # use_fast_accum=True,
        )
        print(f"{scale_result=}")
        ctx.save_for_backward(x_fp8, W_fp8, scale_x, scale_W)
        # ctx.save_for_backward(x, W)
        # print(f"{x.shape=}, {W_fp8.shape=}, {y.shape=}")
        assert y.dtype == torch.float32
        assert y.is_contiguous()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # grad_y: (N, out_features)
        x_fp8, W_fp8, scale_x, scale_W = ctx.saved_tensors  # are in FP8
        grad_y_fp8, scale_y = to_fp8(grad_y)

        # dL/dx = grad_y @ W^T
        grad_x = None
        if ctx.needs_input_grad[0]:
            # grad_x = grad_y.matmul(W)
            grad_x = torch._scaled_mm(
                input=grad_y_fp8,
                mat2=W_fp8.T.contiguous().T,  # ouch
                bias=None,
                out_dtype=torch.float32,
                scale_a=scale_y,
                scale_b=scale_W,
            )
            assert grad_x.is_contiguous()

        # dL/dW = x^T @ grad_y
        grad_W = None
        if ctx.needs_input_grad[1]:
            # grad_W = grad_y.T.matmul(x)
            grad_W = torch._scaled_mm(
                input=grad_y_fp8.T.contiguous(),
                mat2=x_fp8.T.contiguous().T,
                bias=None,
                out_dtype=torch.float32,
                scale_a=scale_y,
                scale_b=scale_x,
            )
            assert grad_W.is_contiguous()

        return grad_x, grad_W


# Convenience module wrapper so it behaves like nn.Linear
class CustomLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        assert bias == False, "Bias not implemented"
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return FP8Linear.apply(x, self.weight)


class CustomLinearFullFP8(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        assert bias == False, "Bias not implemented"
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return LinearFullFP8.apply(x, self.weight)


def test_custom_linear(linear_layer):
    # quick test vs autograd
    # lin = CustomLinear(16, 16, bias=False)
    lin = linear_layer(16, 16, bias=False)
    x = torch.randn(16, 16)
    x[0:15, :] = 0
    x.requires_grad_(True)
    loss = lin(x).pow(2).sum()
    loss.backward()

    # print debug info

    print(f"{lin.weight.grad=}")
    # print(f"{x.grad=}")

    # compare to nn.Linear
    lin2 = torch.nn.Linear(4, 3, bias=False)
    lin2.weight.data = lin.weight.data
    x2 = x.detach().clone().requires_grad_(True)
    loss2 = lin2(x2).pow(2).sum()
    loss2.backward()

    print(f"{lin2.weight.grad=}")
    # print(f"{x2.grad=}")

    # compare tensor metadata
    print(f"{lin.weight.grad.stride()=}")
    print(f"{lin2.weight.grad.stride()=}")
    print(f"{lin.weight.grad.shape=}")
    print(f"{lin2.weight.grad.shape=}")
    print(f"{lin.weight.grad.is_contiguous()=}")
    print(f"{lin2.weight.grad.is_contiguous()=}")

    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-1, atol=5e-2)
    torch.testing.assert_close(lin.weight.grad, lin2.weight.grad, rtol=1e-1, atol=5e-2)


if __name__ == "__main__":
    # test_custom_linear(CustomLinear)
    test_custom_linear(CustomLinearFullFP8)
