#include <cuda_runtime.h>

// 逐元素绝对值 kernel，使用 float4 向量化处理连续内存，提高内存吞吐。
__global__ void abs_kernel(float* __restrict__ output, const float* __restrict__ input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t vec_numel = numel / 4;
    const int64_t vec_stride = stride;

    // 向量化部分
    for (int64_t i = idx; i < vec_numel; i += vec_stride) {
        const float4* in_vec = reinterpret_cast<const float4*>(input) + i;
        float4 val = *in_vec;
        val.x = fabsf(val.x);
        val.y = fabsf(val.y);
        val.z = fabsf(val.z);
        val.w = fabsf(val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    // 尾部元素处理
    for (int64_t i = vec_numel * 4 + idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// launcher: 外部 C 接口，供 binding.cpp 调用
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream)
{
    // 选择 block size 和 grid size
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535; // 限制 grid 维度，kernel 内部 grid-stride 循环处理剩余
    }
    abs_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}
