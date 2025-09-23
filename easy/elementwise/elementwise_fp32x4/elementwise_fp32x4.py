import time
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# ===============================
# Load the CUDA kernel (fp32x4)
# ===============================
lib = load(
    name="elementwise_fp32x4",
    sources=["elementwise_fp32x4.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-gencode", "arch=compute_89,code=sm_89" 
    ],
    extra_cflags=["-std=c++17"],
    verbose=True
)

# ===============================
# Benchmark function
# ===============================
def run_benchmark(func, a, b, out=None, warmup=10, iters=1000):
    if out is not None:
        out.fill_(0)
    # Warmup
    for _ in range(warmup):
        func(a, b, out)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        func(a, b, out)
    torch.cuda.synchronize()
    end = time.time()

    mean_time = (end - start) * 1000 / iters  # ms per iteration
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"fp32x4: {out_val}, mean time: {mean_time:.8f} ms")
    return out, mean_time

# ===============================
# Test different sizes
# ===============================
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]

for S in Ss:
    for K in Ks:
        print("-" * 60)
        print(f"S={S}, K={K}")
        a = torch.randn((S, K), device="cuda", dtype=torch.float32).contiguous()
        b = torch.randn((S, K), device="cuda", dtype=torch.float32).contiguous()
        c = torch.zeros_like(a)
        
        run_benchmark(lib.elementwise_add_f32x4, a, b, c)

        # 验证结果正确性
        if torch.allclose(a+b, c):
            print("Result check: PASS")
        else:
            print("Result check: FAIL")
