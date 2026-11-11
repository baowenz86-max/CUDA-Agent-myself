#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// custom_logical_or: 对两个 float32 输入逐元素比较后逻辑或，输出 bool
// -----------------------------------------------------------------------------
__global__ void logical_or_kernel(const float* __restrict__ x,
                                  const float* __restrict__ y,
                                  bool* __restrict__ out,
                                  long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = gridDim.x * (long long)blockDim.x;
    for (; idx < numel; idx += stride) {
        out[idx] = (x[idx] > 0.5f) || (y[idx] > 0.5f);
    }
}

extern "C" void custom_logical_or_launcher(const float* x,
                                            const float* y,
                                            bool* out,
                                            long long numel,
                                            cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (numel + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;  // 网格上限，实际网格步长循环覆盖所有元素
    logical_or_kernel<<<blocks, block_size, 0, stream>>>(x, y, out, numel);
}

// -----------------------------------------------------------------------------
// custom_cast_float: bool -> float32
// -----------------------------------------------------------------------------
__global__ void cast_float_kernel(const bool* __restrict__ in,
                                  float* __restrict__ out,
                                  long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = gridDim.x * (long long)blockDim.x;
    for (; idx < numel; idx += stride) {
        out[idx] = in[idx] ? 1.0f : 0.0f;
    }
}

extern "C" void custom_cast_float_launcher(const bool* in,
                                            float* out,
                                            long long numel,
                                            cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (numel + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    cast_float_kernel<<<blocks, block_size, 0, stream>>>(in, out, numel);
}

// -----------------------------------------------------------------------------
// custom_mse_loss_mean: 均方误差，输出标量 float32
// 采用两阶段归约：先在 block 内归约，再由单个 block 汇总
// -----------------------------------------------------------------------------
__global__ void mse_loss_block_partial_kernel(const float* __restrict__ input,
                                              const float* __restrict__ target,
                                              float* __restrict__ partial_sums,
                                              long long numel) {
    extern __shared__ float sdata[];
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = gridDim.x * (long long)blockDim.x;
    float sum = 0.0f;
    for (; idx < numel; idx += stride) {
        float diff = input[idx] - target[idx];
        sum += diff * diff;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // block 内归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void mse_loss_final_kernel(const float* __restrict__ partial_sums,
                                      float* __restrict__ output,
                                      int num_partials,
                                      long long total_numel) {
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[0] = sdata[0] / (float)total_numel;
    }
}

extern "C" void custom_mse_loss_mean_launcher(const float* input,
                                               const float* target,
                                               float* output,
                                               long long numel,
                                               cudaStream_t stream) {
    const int block_size = 256;
    // 限制块数，避免过多 partial 数组
    long long blocks = (numel + block_size - 1) / block_size;
    if (blocks > 1024) blocks = 1024;  // 最多 1024 个 partial sum

    // 分配临时存储 partial sums（由 binding 层预先分配？这里直接使用 cudaMalloc）
    // 注意：不能在 launcher 内分配后忘记释放，但 binding 层会用 torch::empty 分配 partial 张量。
    // 此处简化：直接使用 cudaMalloc，但可能造成内存泄漏。更合适的做法是由 binding 传指针。
    // 然而 API 要求 launcher 仅接受原始指针、标量和 stream，因此 partial_sums 缓冲区应由调用者提供。
    // 本文件仅提供 kernel，binding 层负责分配 partial 缓冲区并传入。但我们的 launcher 签名需要 partial 指针参数？
    // 根据契约，custom_mse_loss_mean 的参数只有 input 和 target，输出标量。因此我们需要在 launcher 内部
    // 临时分配 partial 内存，调用 kernel 后释放。使用 cudaMalloc/cudaFree 是安全的，但开销较大。
    // 另一种方案：单 kernel 使用原子累加，但可能会损失精度。这里选择 cudaMalloc 方式。
    float* partial_sums = nullptr;
    cudaMalloc(&partial_sums, blocks * sizeof(float));

    mse_loss_block_partial_kernel<<<blocks, block_size, block_size * sizeof(float), stream>>>(
        input, target, partial_sums, numel);

    // 最终归约：使用单个 block
    mse_loss_final_kernel<<<1, block_size, block_size * sizeof(float), stream>>>(
        partial_sums, output, (int)blocks, numel);

    cudaFree(partial_sums);
}
