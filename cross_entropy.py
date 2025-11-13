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


@torch.compile(
    mode="max-autotune", fullgraph=True
)  # mode="reduce-overhead", fullgraph=True
def _cross_entropy_with_logits_checkpointed(
    embeddings, weights, targets
) -> torch.Tensor:
    return checkpoint.checkpoint(  # type: ignore
        _cross_entropy_with_logits,
        embeddings,
        weights,
        targets,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def cross_entropy_with_logits_by_segment(embeddings, weights, targets):
    """Splits the batch into segments and computes the loss for each segment.

    This lowers memory usage.
    """
    assert embeddings.shape[0] == targets.shape[0]
    assert embeddings.ndim == 2
    batch_size = embeddings.shape[0]
    segment_size = 4096 * 2
    num_segments = (batch_size + segment_size - 1) // segment_size
    total_loss = 0.0
    for i in range(num_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, batch_size)
        segment_embeddings = embeddings[start:end]
        this_segment_size = segment_embeddings.shape[0]
        segment_targets = targets[start:end]
        loss = _cross_entropy_with_logits_checkpointed(
            segment_embeddings, weights, segment_targets
        )
        if this_segment_size < segment_size:
            loss *= this_segment_size / segment_size
        total_loss += loss

    return total_loss / num_segments
