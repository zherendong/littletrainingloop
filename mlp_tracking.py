"""MLP activation tracking for TransformerModel.

Collects per-neuron activation statistics from MLP layers using forward hooks.
Works with both MLP and GrowingMLP without any changes to those classes.

The activation captured is the tensor fed into linear_out, i.e.:
  - non-GLU:  act(linear_in(x))
  - GLU:      act(linear_in(x)) * linear_gate(x)

Output CSV columns
------------------
step        : training step
layer       : transformer layer index (0-based)
block       : block index within a GrowingMLP (always 0 for standard MLP)
neuron      : neuron index within the MLP hidden dim
mean_abs_when_active : mean |activation| among positions where |activation| > threshold
pct_active           : fraction of positions where |activation| > threshold
out_weight_norm      : L2 norm of the neuron's column in the output projection matrix
"""

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformer import TransformerModel


def _hook_factory(store: list, idx: int):
    """Return a pre-hook that captures args[0] (the activation) at slot idx."""
    def hook(module, args):
        store[idx] = args[0].detach()
    return hook


@torch.no_grad()
def collect_mlp_stats(
    model: "TransformerModel",
    inputs: torch.Tensor,
    threshold: float = 0.01,
) -> list[dict]:
    """Collect per-neuron stats from all MLP layers via forward hooks.

    Registers temporary hooks on each linear_out layer, runs the uncompiled
    model._forward, then removes hooks. The compiled training path is
    unaffected.

    Args:
        model: TransformerModel instance.
        inputs: Integer token tensor of shape (batch, seq_len).
        threshold: |activation| threshold for computing pct_active.

    Returns:
        List of dicts with keys: layer, block, neuron, mean_abs, pct_active.
        For standard MLP, block is always 0.
    """
    from transformer import GrowingMLP

    captured: list[torch.Tensor | None] = []
    hook_meta: list[tuple[int, int]] = []  # (layer_idx, block_idx)
    out_weight_norms: list[torch.Tensor] = []  # per-slot, shape (inner_size,)
    hooks = []

    for layer_idx, tb in enumerate(model.transformer_blocks):
        mlp = tb.mlp
        if isinstance(mlp, GrowingMLP):
            for block_idx, block in enumerate(mlp.blocks):
                slot = len(captured)
                captured.append(None)
                hook_meta.append((layer_idx, block_idx))
                out_weight_norms.append(
                    block["linear_out"].weight.float().norm(dim=0).cpu().detach()
                )
                hooks.append(
                    block["linear_out"].register_forward_pre_hook(
                        _hook_factory(captured, slot)
                    )
                )
        else:
            slot = len(captured)
            captured.append(None)
            hook_meta.append((layer_idx, 0))
            out_weight_norms.append(
                mlp.linear_out.weight.float().norm(dim=0).cpu().detach()
            )
            hooks.append(
                mlp.linear_out.register_forward_pre_hook(
                    _hook_factory(captured, slot)
                )
            )

    try:
        model._forward(inputs)
    finally:
        for h in hooks:
            h.remove()

    rows = []
    for slot, acts in enumerate(captured):
        if acts is None:
            continue
        layer_idx, block_idx = hook_meta[slot]
        acts_f = acts.float().view(-1, acts.shape[-1])  # (B*T, inner_size)
        abs_acts = acts_f.abs()
        active_mask = abs_acts > threshold          # (B*T, inner_size)
        pct_active = active_mask.float().mean(dim=0).cpu()
        # mean abs among active positions; fall back to 0 if never active
        active_sum = (abs_acts * active_mask).sum(dim=0)
        active_count = active_mask.sum(dim=0).clamp(min=1)
        mean_abs_when_active = (active_sum / active_count).cpu()
        out_norms = out_weight_norms[slot]
        for neuron_idx in range(len(pct_active)):
            rows.append({
                "layer": layer_idx,
                "block": block_idx,
                "neuron": neuron_idx,
                "mean_abs_when_active": round(mean_abs_when_active[neuron_idx].item(), 6),
                "pct_active": round(pct_active[neuron_idx].item(), 4),
                "out_weight_norm": round(out_norms[neuron_idx].item(), 6),
            })
    return rows


class MLPStatsWriter:
    """Write per-neuron activation stats to a CSV file."""

    FIELDNAMES = ["step", "layer", "block", "neuron", "mean_abs_when_active", "pct_active", "out_weight_norm"]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def write(self, step: int, rows: list[dict]) -> None:
        for row in rows:
            self._writer.writerow({"step": step, **row})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class MLPTracker:
    """Periodically collects MLP activation stats during training.

    Example usage in training setup:
        tracker = MLPTracker(model, "results/mlp_stats.csv", track_every=500)
        # pass tracker.make_track_fn() to training_loop.train as mlp_track_fn
        tracker.close()  # at end of training
    """

    def __init__(
        self,
        model: "TransformerModel",
        output_path: str | Path,
        track_every: int = 500,
        threshold: float = 0.2,
    ):
        self.model = model
        self.writer = MLPStatsWriter(output_path)
        self.track_every = track_every
        self.threshold = threshold

    def maybe_track(self, step: int, inputs: torch.Tensor) -> None:
        if step % self.track_every != 0:
            return
        rows = collect_mlp_stats(self.model, inputs, self.threshold)
        self.writer.write(step, rows)

    def close(self) -> None:
        self.writer.close()
