#include <torch/extension.h>
#include <cuda_runtime.h>

// float4向量化的kernel
__global__ void elementwise_add_f32x4_kernel (const float4 *A, const float4 *B, float4 *C, int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        float4 va = A[idx];
        float4 vb = B[idx];
        C[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}

void elementwise_add_f32x4 (torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.numel();
    int N4 = N / 4;
    int threads = 256;
    int blocks = (N4 + threads - 1) / threads;

    // 主体部分按照float4来进行处理
    elementwise_add_f32x4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(A.data_ptr<float>()),
        reinterpret_cast<const float4*>(B.data_ptr<float>()),
        reinterpret_cast<float4*>(C.data_ptr<float>()),
        N4
    );

    // 尾部处理
    int remain = N % 4;
    if (remain > 0) {
        int start = N - remain;
        auto A_ptr = A.data_ptr<float>();
        auto B_ptr = B.data_ptr<float>();
        auto C_ptr = C.data_ptr<float>();
        for (int i = 0; i < remain; i++) {
            C_ptr[start + i] = A_ptr[start + i] + B_ptr[start + i];
        }
    }
}

// PyTorch绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_f32x4", &elementwise_add_f32x4, "Elementwise Add f32x4 (float4 vectorized)");
}
