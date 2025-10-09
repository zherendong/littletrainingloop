"""
Test script for cross_entropy.py.


Also includes tests for cut cross entropy (linear cross entropy).

"Cut Your Losses in Large-Vocabulary Language Models" https://arxiv.org/abs/2411.09009
"""

import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
import cross_entropy


# make cuda default
if torch.cuda.is_available():
    torch.set_default_device("cuda")


def test_linear_cross_entropy_shape():
    """Smoke test of the linear_cross_entropy function."""
    embeddings = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    projection_weights = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    targets = torch.tensor([0, 1])
    loss = linear_cross_entropy(embeddings, projection_weights, targets)
    assert loss.shape == ()


def compare_linear_cross_entropy(embeddings, projection_weights, targets):
    loss_linear = linear_cross_entropy(
        embeddings,
        projection_weights,
        targets,
        accum_e_fp32=True,
        accum_c_fp32=True,
    )
    loss_linear = loss_linear.to(torch.float32)

    logits = embeddings @ projection_weights.t()
    loss_cross_entropy = torch.nn.functional.cross_entropy(logits, targets)
    loss_cross_entropy = loss_cross_entropy.to(torch.float32)

    print(f"{loss_linear=}, {loss_cross_entropy=}")
    torch.testing.assert_close(loss_linear, loss_cross_entropy, rtol=5e-3, atol=5e-3)


def test_linear_cross_entropy_equal_to_cross_entropy():
    """linear_cross_entropy should be equal to cross_entropy"""
    embeddings = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    projection_weights = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    targets = torch.tensor([0, 1])

    compare_linear_cross_entropy(embeddings, projection_weights, targets)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_equivalence_with_random_data(dtype: torch.dtype):
    embeddings = torch.randn(512, 128, dtype=dtype)  # 512 samples, 128 dimensions
    projection_weights = torch.randn(1000, 128, dtype=dtype)  # 1000 classes
    targets = torch.randint(0, 1000, (512,), dtype=torch.long)

    compare_linear_cross_entropy(embeddings, projection_weights, targets)


def test_segmented_cross_entropy_equivalence(dtype=torch.float32):
    embeddings = torch.randn(
        29000, 128, device="cuda", dtype=dtype
    )  # 512 samples, 128 dimensions
    projection_weights = torch.randn(
        1000, 128, device="cuda", dtype=dtype
    )  # 1000 classes
    targets = torch.randint(0, 1000, (29000,), dtype=torch.long, device="cuda")

    loss1 = cross_entropy.cross_entropy_with_logits_by_segment(
        embeddings, projection_weights, targets
    )
    loss2 = cross_entropy._cross_entropy_with_logits(
        embeddings, projection_weights, targets
    )
    torch.testing.assert_close(loss1, loss2, rtol=5e-3, atol=5e-3)
