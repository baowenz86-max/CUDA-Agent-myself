#include <cuda_runtime.h>

// isin_forward CUDA kernel: 检查 x 中每个元素是否在 test_elements 中
// 使用线性搜索，test_elements 通常较小（1024），足够快
__global__ void isin_kernel(const float* __restrict__ x,
                            const float* __restrict__ test_elements,
                            bool* __restrict__ output,
                            int64_t numel,
                            int64_t test_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = x[i];
        bool found = false;
        for (int64_t j = 0; j < test_size; ++j) {
            if (val == test_elements[j]) {
                found = true;
                break;
            }
        }
        output[i] = found;
    }
}

// 原子操作不需要，直接写 output
// launcher: 由下一阶段 binding.cpp 调用
extern "C" void isin_launcher(const float* x,
                             const float* test_elements,
                             bool* output,
                             int64_t numel,
                             int64_t test_size,
                             cudaStream_t stream) {
    // 参数有效性检查交给 binding 层；这里仅计算启动配置
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    // 限制 blocks 数量，grid-stride 处理大张量
    if (blocks > 65535) blocks = 65535;
    isin_kernel<<<blocks, threads, 0, stream>>>(x, test_elements, output, numel, test_size);
}

// 本文件不包含 torch 依赖，符合 SKILL.md 要求。
// is_complex_forward 无需 GPU kernel，由 binding.cpp 直接实现。