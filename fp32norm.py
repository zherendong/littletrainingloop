"""Normalization layers that keep the computation in fp32."""

import torch
import torch.nn as nn


class FP32LayerNorm(nn.Module):
    def __init__(self, input_size: int, segment_size: int | None = None):
        super(FP32LayerNorm, self).__init__()
        self.input_size = input_size
        self.segment_size = segment_size
        self.norm = nn.LayerNorm(
            segment_size or input_size,
            eps=1e-5,  # supposedly helps with stability
            bias=False,
            dtype=torch.float32,
        )

    def init_weights(self, init_val: float | None = None):
        if init_val is None:
            self.norm.reset_parameters()
        else:
            self.norm.weight.data.fill_(init_val)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if self.segment_size:
            assert self.input_size % self.segment_size == 0
            num_segments = self.input_size // self.segment_size
            orig_shape = x.shape
            x = x.view(-1, num_segments, self.segment_size)
            x = self.norm(x)
            x = x.view(*orig_shape)
        else:
            x = self.norm(x)
        return x.to(input_dtype)


# TODO: turn into FP32 norm
# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def init_weights(self):
#         nn.init.ones_(self.weight)

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         return output * self.weight
