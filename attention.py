"""Attention functions."""

use_flash_attention = True
try:
    import flash_attn
except ImportError:
    print("Could not load flash attention. Falling back to manual attn.")
    use_flash_attention = False
import torch
from torch.utils.flop_counter import register_flop_formula


def attention_fn(q, k, v, use_flash: bool = True):
    if use_flash and use_flash_attention:
        batch_size, sequence_length_q, num_heads_kv, q_per_kv, head_dim = q.shape
        orig_dtype = q.dtype
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        num_heads_q = num_heads_kv * q_per_kv
        q = q.view(batch_size, sequence_length_q, num_heads_q, head_dim)
        q_per_kv = num_heads_q // num_heads_kv
        sequence_length_kv = k.shape[1]
        k = k.view(batch_size, sequence_length_kv, num_heads_kv, head_dim)
        v = v.view(batch_size, sequence_length_kv, num_heads_kv, head_dim)
        res: torch.Tensor = flash_attn.flash_attn_func(q, k, v, causal=True)  # type: ignore
        return res.view(
            batch_size, sequence_length_q, num_heads_kv, q_per_kv, head_dim
        ).to(orig_dtype)
    batch_size, sequence_length_q, num_heads_kv, q_per_kv, head_dim = q.shape
    num_heads_q = num_heads_kv * q_per_kv
    sequence_length_kv = k.shape[1]
    # Names for einsum: b, t, h, q, d
    q *= 1 / head_dim**0.5
    scores = torch.einsum("bthqd,bThd->btThq", q, k)
    assert (
        sequence_length_q == sequence_length_kv
    ), "Need to test the triangle code path if this doesn't hold."
    causal_mask = torch.triu(
        torch.ones(
            sequence_length_q, sequence_length_kv, device=q.device, dtype=q.dtype
        )
        * float("-inf"),
        diagonal=1,
    ).view(1, sequence_length_q, sequence_length_kv, 1, 1)
    scores += causal_mask
    probs = torch.softmax(scores, dim=2)
    out = torch.einsum("btThq,bThd->bthqd", probs, v)
    return out


if use_flash_attention:

    @register_flop_formula(torch.ops.flash_attn._flash_attn_forward)
    def attention_fn_flop_formula(q_shape, k_shape, v_shape, *args, **kwargs):
        # del out_shape
        # print(
        #     f"Computing flops for flash attention with {q_shape=}, {k_shape=}, {v_shape=}"
        # )
        # print(f"args: {args}, kwargs: {kwargs}")

        # Assuming that none of the fancy parameters are set. Assuming causal=True.
        batch_size, sequence_length_q, num_heads_q, head_dim = q_shape
        batch_size, sequence_length_kv, _, head_dim = k_shape
        batch_size, sequence_length_kv, _, head_dim = v_shape
        qk_flops = (
            batch_size
            * num_heads_q
            * sequence_length_q
            * head_dim
            * sequence_length_kv
            * 2  # multiply-add
            // 2  # because causal
        )
        softmax_flops = (
            batch_size * num_heads_q * sequence_length_q * sequence_length_kv * 3 // 2
        )
        qv_flops = (
            batch_size
            * num_heads_q
            * sequence_length_q
            * sequence_length_kv
            * head_dim
            * 2  # multiply-add
            // 2  # because causal
        )
        return qk_flops + softmax_flops + qv_flops

    @register_flop_formula(torch.ops.flash_attn._flash_attn_backward)
    def attention_fn_bwd_flop_formula(
        dout_shape,
        q_shape,
        k_shape,
        v_shape,
        *args,
        **kwargs,
    ):
        # print(
        #     f"Computing flops for flash attention backward with {dout_shape=}, {q_shape=}, {k_shape=}, {v_shape=}"
        # )
        return attention_fn_flop_formula(q_shape, k_shape, v_shape) * 2
