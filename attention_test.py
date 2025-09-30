import torch
import attention


def test_attention_fn():
    """Smoke test for the attention_fn function"""
    q = torch.randn(1, 3, 2, 4, 32, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16)
    v = torch.randn(1, 3, 2, 32, dtype=torch.bfloat16)
    y = attention.attention_fn(q, k, v)
    assert y.shape == (1, 3, 2, 4, 32)
    assert not torch.isnan(y).any()


def test_flash_nonflash_equivalence():
    """Test that flash and non-flash attention give the same results"""
    if not torch.cuda.is_available():
        return
    q = torch.normal(
        mean=0, std=1, size=(1, 3, 2, 4, 32), dtype=torch.bfloat16, device="cuda"
    )
    k = torch.normal(
        mean=0, std=1, size=(1, 3, 2, 32), dtype=torch.bfloat16, device="cuda"
    )
    v = torch.normal(
        mean=0, std=1, size=(1, 3, 2, 32), dtype=torch.bfloat16, device="cuda"
    )
    y_flash = attention.attention_fn(q, k, v, use_flash=True)
    y_nonflash = attention.attention_fn(q, k, v, use_flash=False)
    torch.testing.assert_close(y_flash, y_nonflash, rtol=1e-2, atol=1e-2)
