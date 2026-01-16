"""
GrowingMLP: An MLP that can be extended with new blocks during training.

Each block is a slice of the MLP with its own key-value pairs.
Blocks can be added at specific training steps with fresh warmup schedules.
"""

import torch
import torch.nn as nn
from typing import Iterator

import sys

sys.path.insert(0, "../..")
import fp32norm
import initialization


class MLPBlock(nn.Module):
    """A single block of key-value pairs in the MLP.

    Each block has its own norm layer for clean learning dynamics.
    New blocks don't need to worry about where a shared norm has moved.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        input_size: int,
        output_size: int,
        block_size: int,  # number of KV pairs in this block
        nonlinearity: str = "swish",
        glu: bool = True,
        pairwise_cancelling_init: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.glu = glu
        self.pairwise_cancelling_init = pairwise_cancelling_init
        self.nonlinearity = nonlinearity

        # Each block has its own norm
        self.norm = fp32norm.FP32LayerNorm(input_size)

        # Keys: project input to block_size activations
        self.linear_in = nn.Linear(input_size, block_size, bias=False, dtype=dtype)

        # GLU gate (optional)
        if glu:
            self.linear_glu = nn.Linear(input_size, block_size, bias=False, dtype=dtype)

        # Nonlinearity
        if nonlinearity == "relu":
            self.nonlinear_fn = nn.ReLU()
        elif nonlinearity == "gelu":
            self.nonlinear_fn = nn.GELU()
        elif nonlinearity == "swish":
            self.nonlinear_fn = nn.SiLU()
        else:
            raise NotImplementedError(f"Unknown nonlinearity: {nonlinearity}")

        # Values: project block_size back to output_size
        self.linear_out = nn.Linear(block_size, output_size, bias=False, dtype=dtype)

    def init_weights(self):
        """Initialize with pairwise cancelling init."""
        self.norm.init_weights()
        initialization.init_linear(
            self.linear_in,
            activation=self.nonlinearity,
            pairwise_mode="equal" if self.pairwise_cancelling_init else None,
        )
        if self.glu:
            initialization.init_linear(
                self.linear_glu,
                activation="linear",
                pairwise_mode="equal" if self.pairwise_cancelling_init else None,
            )
        initialization.init_linear(
            self.linear_out,
            pairwise_mode="opposing" if self.pairwise_cancelling_init else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Each block normalizes its own input."""
        x_norm = self.norm(x)
        if self.glu:
            keys = self.linear_in(x_norm)
            gates = self.linear_glu(x_norm)
            activations = self.nonlinear_fn(keys) * gates
            return self.linear_out(activations)
        else:
            keys = self.linear_in(x_norm)
            activations = self.nonlinear_fn(keys)
            return self.linear_out(activations)


class GrowingMLP(nn.Module):
    """MLP that can grow by adding new blocks during training.

    Each block has its own norm layer for clean learning dynamics.
    Each block contributes additively to the output.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        input_size: int,
        output_size: int | None = None,
        block_size: int = 512,
        initial_blocks: int = 1,
        nonlinearity: str = "swish",
        glu: bool = True,
        pairwise_cancelling_init: bool = True,
        copy_most_active_init: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.block_size = block_size
        self.nonlinearity = nonlinearity
        self.glu = glu
        self.pairwise_cancelling_init = pairwise_cancelling_init
        self.copy_most_active_init = copy_most_active_init
        self.dtype = dtype

        # Initial blocks (each has its own norm)
        self.blocks = nn.ModuleList()
        for _ in range(initial_blocks):
            self.blocks.append(self._make_block())

    def _make_block(self) -> MLPBlock:
        """Create a new block with current config."""
        return MLPBlock(
            dtype=self.dtype,
            input_size=self.input_size,
            output_size=self.output_size,
            block_size=self.block_size,
            nonlinearity=self.nonlinearity,
            glu=self.glu,
            pairwise_cancelling_init=self.pairwise_cancelling_init,
        )

    def init_weights(self):
        """Initialize all blocks."""
        for block in self.blocks:
            assert isinstance(block, MLPBlock)
            block.init_weights()

    def add_block(self) -> Iterator[nn.Parameter]:
        """Add a new block and return its parameters (for optimizer)."""
        last_block = self.blocks[-1]
        new_block = self._make_block()
        new_block.init_weights()
        self.blocks.append(new_block)

        if self.copy_most_active_init:
            # TODO: identify the most commonly activated keys and split them
            # for old, new in zip(last_block.parameters(), new_block.parameters()):
            #     new.data.copy_(old.data * 0.2)
            print("Copying last block's weights")
            # print on which devices these tensors are
            new_block.linear_in.weight.data = (
                0.5 * new_block.linear_in.weight.data
                + 0.5 * last_block.linear_in.weight.data
            )

        return new_block.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks (each block normalizes its own input)."""
        # Sum contributions from all blocks
        output = self.blocks[0](x)
        for block in self.blocks[1:]:
            output = output + block(x)
        return output

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def total_inner_size(self) -> int:
        """Total number of KV pairs across all blocks."""
        return self.num_blocks * self.block_size
