import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    Format,
    DelayedScaling,
    Float8BlockScaling,
)

# Basic sanity: pin to one GPU if you want
assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda")

# --- FP8 recipes ---
fp8_recipe_hybrid = DelayedScaling(
    fp8_format=Format.HYBRID,  # E4M3 (fwd) / E5M2 (bwd)
    amax_history_len=16,
    amax_compute_algo="max",
)

fp8_recipe_block = Float8BlockScaling(fp8_format=Format.E4M3)


def tflops_linear(M, K, N, ms):
    # GEMM flops = 2*M*K*N, convert to TFLOP/s
    flops = 2.0 * M * K * N
    return (flops / (ms * 1e-3)) / 1e12


def delay_compute():
    # do a big operation to hide CPU scheduling latency
    a = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
    a @ b


@torch.inference_mode()
def bench_forward_linear_te(
    batch=128,
    M=2048,
    K=8192,
    N=2048,
    iters=50,
    warmup=3,
    recipe=fp8_recipe_hybrid,
    bias=False,
):
    # TE linear prefers BF16 inputs/weights when used with fp8_autocast
    # with te.fp8_model_init(enabled=True, preserve_high_precision_init_val=False):
    layer = te.Linear(K, N, bias=bias, device=device, params_dtype=torch.bfloat16)
    x = torch.rand(batch, M, K, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = layer(x)
    torch.cuda.synchronize()

    delay_compute()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    start.record()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        for _ in range(iters):
            y = layer(x)
    end.record()
    end.synchronize()
    times_ms.append(start.elapsed_time(end))

    avg = sum(times_ms) / len(times_ms) / iters
    return {
        "impl": "TE.Linear FP8",
        "recipe": type(recipe).__name__,
        "shape": (M, K, N),
        "iters": iters,
        "avg_ms": avg,
        "avg_tflops": tflops_linear(M, K, N, avg) * batch,
    }


def bench_forward_backward_linear_te(
    batch=128,
    M=2048,
    K=8192,
    N=2048,
    iters=50,
    warmup=3,
    recipe=fp8_recipe_hybrid,
    bias=False,
):
    layer = te.Linear(K, N, bias=bias, device=device).to(torch.bfloat16)
    x = torch.rand(batch, M, K, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Simple scalar loss to drive backward
    def step():
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = layer(x)
            loss = y.sum() / (M * N)
        loss.backward()

    # Warmup
    for _ in range(warmup):
        step()
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
    torch.cuda.synchronize()

    delay_compute()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    start.record()
    for _ in range(iters):
        step()
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
    end.record()
    end.synchronize()
    times_ms.append(start.elapsed_time(end))

    avg = sum(times_ms) / len(times_ms) / iters
    # For fwd+bwd, a common quick proxy is ~3x GEMM flops (fwd + dA + dB).
    # This is a heuristic; true flops depend on bias, activation scaling, etc.
    flops_multiplier = 3.0

    def tf(ms):
        return flops_multiplier * tflops_linear(M, K, N, ms)

    return {
        "impl": "TE.Linear FP8 (fwd+bwd)",
        "recipe": type(recipe).__name__,
        "shape": (M, K, N),
        "iters": iters,
        "avg_ms": avg,
        "avg_tflops_equiv": tf(avg) * batch,
    }


@torch.inference_mode()
def bench_forward_baseline_bf16(
    batch=128,
    M=2048,
    K=8192,
    N=2048,
    iters=50,
    warmup=3,
    bias=False,
):
    # Baseline using plain PyTorch BF16 Linear for comparison
    layer = torch.nn.Linear(K, N, bias=bias, device=device, dtype=torch.bfloat16)
    x = torch.rand(batch, M, K, device=device, dtype=torch.bfloat16)

    for _ in range(warmup):
        y = layer(x)
    torch.cuda.synchronize()

    delay_compute()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    start.record()
    for _ in range(iters):
        y = layer(x)
    end.record()
    end.synchronize()
    times_ms.append(start.elapsed_time(end))

    avg = sum(times_ms) / len(times_ms) / iters
    return {
        "impl": "torch.nn.Linear BF16",
        "shape": (M, K, N),
        "iters": iters,
        "avg_ms": avg,
        "avg_tflops": tflops_linear(M, K, N, avg) * batch,
    }


def main():
    torch.backends.cudnn.benchmark = True  # harmless here
    torch.cuda.synchronize()

    # Choose matmul sizes that are multiples of 64 for best FP8 tensorcore usage
    cases = [
        # (1024, 768, 768),
        # (2048, 1536, 1536),
        (2048, 2048, 2048),
    ]

    results = []
    for M, K, N in cases:
        results.append(bench_forward_baseline_bf16(M, K, N))
        results.append(bench_forward_linear_te(M, K, N, recipe=fp8_recipe_block))
        results.append(
            bench_forward_backward_linear_te(M, K, N, recipe=fp8_recipe_hybrid)
        )

    # Pretty print
    from pprint import pprint

    print("\n=== Results ===")
    for r in results:
        pprint(r)


if __name__ == "__main__":
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    main()
