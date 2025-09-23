# CUDA入门

本文为入门CUDA的简单知识点的整理。

## 线程模型

GPU上的计算由大量的线程组成，因此有了三层的抽象概念——线程、线程块和线程块网络。

> thread -> block -> grid

计算最小的执行单元是一个线程，很多线程组成一个线程块，很多线程块组成一个线程块网格。

CUDA中有内置变量帮助我们识别线程在计算当中的位置，例如：

- `threadIdx.x`表示线程在一个线程块内的一维编号
- `blockIdx.x`表示当前的线程块在线程块网格中的编号
- `blockDim.x`表示在一个线程块中有多少个线程
- `gridDim.x`表示一个线程块网络中有多少个线程块
- `blockIdx.x * blockDim.x + threadIdx.x`计算出一个线程在全局中的唯一索引

## 内存模型

CUDA的内存模型和计算机的存储系统设计思想一样，为了速度采用分层的思想。

- 寄存器：最快，每个线程私有
- 共享内存：一个线程块中的所有线程共享，速度快，适合块内通信
- 全局内存：最慢，但是所有的线程都可以访问，容量大

## 核函数

核函数（Kernel Funtion）就是在GPU上运行的函数，通常用`__global__`声明，例如：

```c
__global__ void kernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

在调用时应遵循如下格式：

```c
kernel<<<blocks, threads_per_block>>>(A, B, C, N);
```

## Pytorch的C++/CUDA拓展

CUDA内核和PyTorch中的张量结合，需要用到PyTorch的C++ API。

## PyBind

PyBind11是一个C++11的库，用来把C++写的类/函数导出为Python模块。

例如，如果我们使用C++写了一个加法函数：

```cpp
int add (int x, int y) {
    return x + y
}
```

如果我们要在Python中使用，必须要借助PyBind，应该这样写：

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

// 暴露给 Python
PYBIND11_MODULE(example, m) {
    m.def("add", &add, "A function that adds two numbers");
}

```

## PyTorch的C++拓展

PyTorch的C++拓展就是基于PyBind11的，需要我们使用封装好的库：

```cpp
#include <torch/extension.h>
```

然后导入：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Elementwise Add (CUDA)");
}
```

需要深入理解C++与Python的语法特性，充分发挥两者的优势，在实际开发中实现高效的优化。
