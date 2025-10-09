import torch
from torch.utils import checkpoint

# Ignore_index is a magic number for cross entropy loss
# when a target token is set to this value, we ignore the
# loss for this token.
cross_entropy_ignore_index = -100


def _cross_entropy_with_logits(embeddings, weights, targets):
    logits = embeddings @ weights.t()
    loss = torch.nn.functional.cross_entropy(
        logits,
        targets,
        ignore_index=cross_entropy_ignore_index,
    )
    return loss


@torch.compile(mode="max-autotune", fullgraph=True)
def cross_entropy_with_logits_by_segment(embeddings, weights, targets):
    """Splits the batch into segments and computes the loss for each segment.

    This lowers memory usage.
    """
    assert embeddings.shape[0] == targets.shape[0]
    assert embeddings.ndim == 2
    batch_size = embeddings.shape[0]
    segment_size = 4096
    num_segments = (batch_size + segment_size - 1) // segment_size
    losses = torch.zeros(num_segments, device=embeddings.device, dtype=torch.float32)
    for i in range(num_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, batch_size)
        segment_embeddings = embeddings[start:end]
        segment_targets = targets[start:end]
        losses[i] = checkpoint.checkpoint(  # type: ignore
            _cross_entropy_with_logits,
            segment_embeddings,
            weights,
            segment_targets,
            use_reentrant=False,
        )
    return losses.mean()
