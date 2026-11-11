#include <cuda_runtime.h>

// 拼接 kernel：将 x1 和 x2 沿 dim=1 拼接
// 输入均为 contiguous，形状 [N, C1, H, W] 和 [N, C2, H, W]，输出 [N, C1+C2, H, W]
// 线程数等于输出元素总数，每个线程负责复制一个元素
template <typename scalar_t>
__global__ void custom_cat_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    int64_t total_elements,
    int64_t C1,
    int64_t C2,
    int64_t HW,
    int64_t C_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // 分解输出索引：n, c_out, hw
    int64_t n = idx / (C_out * HW);
    int64_t rem = idx % (C_out * HW);
    int64_t c_out = rem / HW;
    int64_t hw = rem % HW;

    if (c_out < C1) {
        // 属于 x1，计算 x1 中的线性索引
        int64_t src_idx = n * (C1 * HW) + c_out * HW + hw;
        output[idx] = x1[src_idx];
    } else {
        // 属于 x2
        int64_t c2 = c_out - C1;
        int64_t src_idx = n * (C2 * HW) + c2 * HW + hw;
        output[idx] = x2[src_idx];
    }
}

// L1 loss kernel：计算所有元素 |input - target| 的和，使用 double 累加降低舍入误差
// block 内归约，最后块内总和用 atomicAdd 加到全局 double 指针
// 需要使用动态共享内存存储 block 部分和，block 大小固定为 256
template <typename scalar_t>
__global__ void custom_l1_loss_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ target,
    int64_t total_elements,
    double* __restrict__ global_sum) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    int64_t stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    // grid-stride loop
    for (int64_t i = idx; i < total_elements; i += stride) {
        double diff = static_cast<double>(input[i]) - static_cast<double>(target[i]);
        local_sum += fabs(diff);
    }

    // 写入共享内存，归约
    sdata[tid] = local_sum;
    __syncthreads();

    // 树形归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 块内总和由线程0原子加到全局
    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// launcher: custom_cat
// 参数：输出指针、x1指针、x2指针、总元素数、C1、C2、HW、C_out、stream
extern "C" void custom_cat_launcher(
    float* output,
    const float* x1,
    const float* x2,
    int64_t total_elements,
    int64_t C1,
    int64_t C2,
    int64_t HW,
    int64_t C_out,
    cudaStream_t stream) {

    const int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    // 限制 grid 大小（虽然 total_elements 很大，但 blocks 可能超过 2^31-1？这里 2^31-1 足够，但为了安全使用 int64）
    // 实际 GPU 最大 grid dim 为 2^31-1，total_elements 约 2e8，blocks 约 8e5，远小于上限
    custom_cat_kernel<float><<<blocks, threads, 0, stream>>>(
        output, x1, x2, total_elements, C1, C2, HW, C_out);
}

// launcher: custom_l1_loss
// 参数：输入、目标、总元素数、输出标量指针（host 可访问的 double？不行，需要输出 float），
// 但我们先在 device 上计算 double sum，再复制到输出 float 指针
// 这里提供两个 launcher：一个返回 double 指针，但契约要求输出 float tensor，所以在 binding 层分配 device double 临时空间，再转换。
// 为了简单，launcher 接受 float* output（指向标量 tensor），内部先算 double sum，最后写入 float。
extern "C" void custom_l1_loss_launcher(
    const float* input,
    const float* target,
    int64_t total_elements,
    float* output,
    cudaStream_t stream) {
    
    // 在 device 上分配 double 临时空间用于原子累加
    double* device_sum;
    cudaMalloc(&device_sum, sizeof(double));
    cudaMemset(device_sum, 0, sizeof(double));

    const int threads = 256;
    // 选择合适的 grid 大小，平衡并行度和原子竞争
    // 使用足够的块覆盖数据，但不要太多；这里使用 (total_elements + threads*4 -1)/(threads*4)？
    // 通常每个线程处理多个元素，grid 大小根据 SM 数量调整，这里简单使用覆盖所有元素但不超过 65535* 的数量
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // 限制块数，让 grid-stride 循环处理
    size_t shared_mem_size = threads * sizeof(double);

    custom_l1_loss_kernel<float><<<blocks, threads, shared_mem_size, stream>>>(
        input, target, total_elements, device_sum);

    // 将 double sum 复制到临时 float，再输出（或者直接用 kernel 将 double 转 float 写入 output）
    // 这里简单 cudaMemcpy D2D 不行，因为 output 是 float，需要转换。
    // 可以启动一个小 kernel 将 device_sum 转换为 float 写入 output。
    // 但为了减少 kernel，也可以让自定义 kernel 直接 atomicAdd 到 float？精度可能不够。
    // 我们使用一个简单 kernel 完成转换。
    // 但是为了简洁，这里省略转换，由 binding 层处理？不行，launcher 需要提供完整功能。
    // 因此加一个 kernel：
    // 但 .cu 文件应只包含必要 kernel，所以在此文件中定义转换 kernel。
    // 不过更简单：在 custom_l1_loss_kernel 最后，让 tid==0 的线程同时写 output，但需要先计算总和，所有块完成后才能写。
    // 不能保证同步。所以必须在 launcher 中分开步骤。
    // 为了不改动 binding 设计，我们在 launcher 中调用另一个 kernel 将 double 转为 float。
    // 定义转换 kernel：
}

// 转换 kernel：将 device double 转换为 float 写入 output
template <typename scalar_t>
__global__ void double_to_float_kernel(const double* src, float* dst) {
    if (threadIdx.x == 0) {
        dst[0] = static_cast<float>(src[0]);
    }
}

// 修改 launcher 定义
// 注意：函数定义不能在建一半，需要重新整理。下面是完整实现。

// 重新定义 custom_l1_loss_launcher (已包含在上面的不完全定义中，需要修正)
// 为了正确，我们重新给出整个文件内容，忽略之前的不完整代码。
