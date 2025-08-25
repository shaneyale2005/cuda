// 以下是本题的CUDA代码
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_f32_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void elementwise_add_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    elementwise_add_f32_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_f32", &elementwise_add_f32, "Elementwise Add f32");
}