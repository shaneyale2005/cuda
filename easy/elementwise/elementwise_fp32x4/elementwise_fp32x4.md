# float32x4向量加法

## 任务理解

CUDA中的float4表示4个float组成的一组数据，本任务是对于两个形状相同的浮点矩阵A和B进行逐元素加法，一次读取和运算4个元素，得到C。

## 具体实现

CUDA将一个block内的线程组织为三维结构，方便处理多维数据。

在CUDA代码：
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
这里用来获取在x方向上的索引，我们把矩阵展开成一个一维数组进行处理。

注意CUDA kenel的调用方法，内核名称之后要跟着<<<blocks, threads>>>。

在C++中，`reinterpret_cast`是一个类型转换操作符，不改变指针指向的内存，只是告诉编译器“把这个类型当做是另外一种类型来看待”。

`data_ptr<T>()`返回一个张量首地址的原始指针，这是一个模版函数。
