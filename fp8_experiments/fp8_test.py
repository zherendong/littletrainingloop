import torch
import deep_gemm


f32_type = torch.float32
bf16_type = torch.bfloat16
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2


def bench(fn, num_warmups: int = 3, num_tests: int = 10, high_precision: bool = True):
    """Benchmark a function; return time in us."""
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float32, device="cuda")
        y = torch.randn((8192, 8192), dtype=torch.float32, device="cuda")
        for _ in range(5):
            x @ y

    # Testing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_tests):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests * 1e3


def test_scaled_mm(m, k, n):
    device = "cuda"
    mat1 = torch.randn(m, k, device=device, dtype=torch.bfloat16).to(e4m3_type)
    mat2 = torch.randn(n, k, device=device, dtype=torch.bfloat16).to(e4m3_type)
    scale_a = torch.ones((m, 1), device=device)
    scale_b = torch.ones((1, n), device=device)
    # scale_a = torch.tensor(1.0, device=device)
    # scale_b = torch.tensor(1.0, device=device)

    def test_func():
        output = torch._scaled_mm(
            input=mat1,
            mat2=mat2.T,
            bias=None,
            out_dtype=bf16_type,
            scale_a=scale_a,
            scale_b=scale_b,
        )
        return output

    t = bench(test_func, high_precision=True)
    # flops = 2 * m * k * n
    # print(f"{flops / t * 1e6 / 1e12:.2f} TFLOPS")
    # print(f"{t=}us")
    return t


def test_cublas(m, k, n):
    device = "cuda"
    mat1 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    mat2 = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    def test_func():
        output = mat1 @ mat2.t()

    t = bench(test_func, high_precision=True)
    # flops = 2 * m * k * n
    # print(f"{flops / t * 1e6 / 1e12:.2f} TFLOPS")
    # print(f"{t=}us")
    return t


def run_tests():
    shapes = [
        (4096 * 4, 1024, 8192),
        (4096 * 64, 4096, 4096),
        (4096, 32768, 512),
    ]
    for m, k, n in shapes:
        scaled_mm_time_us = test_scaled_mm(m, k, n)
        cublas_time_us = test_cublas(m, k, n)
        print(f"m={m}, k={k}, n={n}")
        print(
            f" > {2 * m * k * n / scaled_mm_time_us * 1e6 / 1e12:.2f} TFLOPS for scaled_mm"
        )
        print(f" > {2 * m * k * n / cublas_time_us * 1e6 / 1e12:.2f} TFLOPS for cublas")
        print(f" > Scaled MM: {scaled_mm_time_us:.2f} us")
        print(f" > BF16 cuBLAS: {cublas_time_us:.2f} us")
        print(f" > Scaled MM / cuBLAS: {scaled_mm_time_us / cublas_time_us:.2f}x")


def scale_experiments(m=2, k=16, n=16):
    device = "cuda"
    mat1 = torch.randn(m, k, device=device, dtype=torch.bfloat16).to(e4m3_type)
    mat2 = torch.randn(n, k, device=device, dtype=torch.bfloat16).to(e4m3_type)
    scale_a = torch.ones((m, 1), device=device)
    scale_b = torch.ones((1, n), device=device)
    # scale_a = torch.tensor(1.0, device=device)
    # scale_b = torch.tensor(1.0, device=device)

    def test_func(scale_a, scale_b):
        output = torch._scaled_mm(
            input=mat1,
            mat2=mat2.T,
            bias=None,
            out_dtype=bf16_type,
            scale_a=scale_a,
            scale_b=scale_b,
        )
        print(f"{output[:8, :8]=}")

    test_func(scale_a, scale_b)
    scale_a = scale_a * 2
    test_func(scale_a, scale_b)
    scale_b[0, 0] = 1 / 2
    test_func(scale_a, scale_b)


if __name__ == "__main__":
    scale_experiments()
    run_tests()

# def test_te():

#     import transformer_engine.pytorch as te
#     from transformer_engine.common.recipe import (
#         Format,
#         DelayedScaling,
#         MXFP8BlockScaling,
#         NVFP4BlockScaling,
#     )

#     fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
#     fp8_recipe = DelayedScaling(
#         fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
#     )
#     mxfp8_format = Format.E4M3  # E4M3 used everywhere
#     mxfp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)
#     nvfp4_recipe = NVFP4BlockScaling()

#     torch.manual_seed(12345)

#     my_linear = te.Linear(768, 768, bias=True)

#     inp = torch.rand((1024, 768)).cuda()

#     with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#         out_fp8 = my_linear(inp)

#     print(out_fp8)
#     assert False


# def test_deep_gemm():
#     device = "cuda"
#     # output = deep_gemm.scaled_mm(
#     #     torch.randn(16, 16, device=device, dtype=f32_type).to(e4m3_type),
#     #     torch.randn(16, 16, device=device).to(e4m3_type).t(),
#     #     # bias=torch.randn(16, device=device).to(bf16_type),
#     #     out_dtype=e4m3_type,
#     #     scale_a=torch.tensor(1.0, device=device),
#     #     scale_b=torch.tensor(1.0, device=device),
#     # )
#     output = deep_gemm.fp8_gemm_nt(
#         a=(torch.randn(16, 16, device=device, dtype=f32_type).to(e4m3_type), 1.0),
#         b=torch.randn(16, 16, device=device, dtype=f32_type).to(e4m3_type),
#         # torch.randn(16, 16, device=device).to(e4m3_type).t(),
#         d=torch.tensor(1.0, device=device),
#         disable_ue8m0_cast=True,
#         # torch.tensor(1.0, device=device),
#     )
#     print(output)
#     assert False
