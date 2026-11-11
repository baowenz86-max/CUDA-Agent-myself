#include <cuda_runtime.h>

// 逐元素比较 kernel: out[i] = (in[i] >= threshold)
__global__ void ge_forward_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    int64_t numel,
    float threshold
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = (input[idx] >= threshold);
    }
}

// C 接口 launcher，供 binding.cpp 调用
extern "C" void custom_ge_forward_launcher(
    const float* input,
    bool* output,
    int64_t numel,
    float threshold,
    cudaStream_t stream
) {
    if (numel == 0) {
        return;
    }
    const int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    // grid 上限保护，使用 grid-stride loop 处理超大张量
    if (blocks > 65535) {
        blocks = 65535;
    }
    ge_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, threshold);
}
