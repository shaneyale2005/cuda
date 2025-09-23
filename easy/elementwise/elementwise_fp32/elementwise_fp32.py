# 测试代码如下
import time
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# 编译并加载 CUDA 扩展
lib = load(
    name="elementwise_lib",
    sources=["elementwise_fp32.cu"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-std=c++17"],
    verbose=True
)

def run_benchmark(func, a, b, out=None, iters=1000):
    # warmup
    for _ in range(10):
        func(a, b, out)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        func(a, b, out)
    torch.cuda.synchronize()
    end = time.time()

    mean_time = (end - start) * 1000 / iters
    print(f"Time per call: {mean_time:.6f} ms")

# 测试数据
N = 1024 * 1024  # 一百万个元素
a = torch.randn(N, device="cuda", dtype=torch.float32)
b = torch.randn(N, device="cuda", dtype=torch.float32)
c = torch.zeros_like(a)

# 跑自定义 kernel
run_benchmark(lib.elementwise_add_f32, a, b, c)

# 跑 PyTorch 内置加法
run_benchmark(lambda x,y,z: torch.add(x,y,out=z), a, b, c)

# 对比结果
print("验证结果是否一致:", torch.allclose(a+b, c))
