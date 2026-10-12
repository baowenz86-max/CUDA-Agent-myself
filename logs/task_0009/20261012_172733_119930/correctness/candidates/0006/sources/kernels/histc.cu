#include <cuda_runtime.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

// Kernel for histogram
// Each thread processes multiple elements using grid-stride loop
__global__ void histc_kernel(
    const float* __restrict__ input,
    long long num_elements,
    int bins,
    float min_val,
    float max_val,
    float* __restrict__ output) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 计算 bin 宽度
    float bin_width = (max_val - min_val) / (float)bins;

    for (long long i = idx; i < num_elements; i += stride) {
        float val = input[i];
        // 忽略 NaN 和范围外值
        if (isnan(val) || val < min_val || val > max_val) {
            continue;
        }
        // 计算 bin 索引
        int bin = (int)((val - min_val) / bin_width);
        // 处理 val == max_val 的情况
        if (bin >= bins) bin = bins - 1;
        // 原子加
        atomicAdd(&output[bin], 1.0f);
    }
}

// Host launcher
extern "C" void histc_forward_launcher(
    const float* input,
    long long num_elements,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream) {

    int block_size = 256;
    long long desired_blocks = (num_elements + block_size - 1) / block_size;
    int max_blocks = 65535;
    int blocks = (int)min(desired_blocks, (long long)max_blocks);

    histc_kernel<<<blocks, block_size, 0, stream>>>(
        input, num_elements, bins, min_val, max_val, output);
}
