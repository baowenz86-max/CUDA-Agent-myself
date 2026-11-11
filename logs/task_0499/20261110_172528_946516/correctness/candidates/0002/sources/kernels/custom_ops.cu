#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
    } \
} while (0)

// 通用逐元素 clamp kernel，支持任意 stride 的非连续张量
// 假设输入所有维度的大小存储在 sizes 中，步长存储在 strides 中，共 ndim 维
// 线程索引对应线性输出索引，通过除法和取模计算多维坐标，再计算输入线性偏移
__global__ void clamp_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int ndim,
    int64_t numel,
    float min_val,
    float max_val
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int64_t remaining = idx;
    int64_t input_offset = 0;
    // 计算多维索引并累加输入偏移
    #pragma unroll
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t dim_size = sizes[d];
        int64_t coord = remaining % dim_size;
        remaining /= dim_size;
        input_offset += coord * strides[d];
    }

    float val = input[input_offset];
    val = fminf(fmaxf(val, min_val), max_val);
    output[input_offset] = val;
}

// argmin kernel: 沿 dim 维度求第一个最小值的索引
// output 形状为原形状去掉 dim 维，用 output_sizes 和 output_strides 描述
// 每个线程处理一个输出元素，遍历 dim 维度
// 注意：输入 strides 和 sizes 为原始维度信息
// output 为 int64_t 类型，输出填充索引值
__global__ void argmin_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    const int64_t* __restrict__ input_sizes,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_sizes,
    const int64_t* __restrict__ output_strides,
    int ndim,
    int dim,
    int64_t output_numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= output_numel) return;

    // 计算输出元素的多维坐标（不含 dim 维度）
    int64_t remaining = idx;
    int64_t out_coords[8]; // 最多支持 8 维，足够
    int64_t out_offset = 0;
    for (int d = ndim - 1, od = ndim - 2; d >= 0; --d) {
        if (d == dim) continue;
        int64_t dim_size = output_sizes[od];
        int64_t coord = remaining % dim_size;
        remaining /= dim_size;
        out_coords[od] = coord;
        out_offset += coord * output_strides[od];
        --od;
    }

    // 计算输入基地址偏移（沿 dim 维度之前的坐标部分）
    int64_t base_input_offset = 0;
    int64_t out_coord_idx = 0;
    for (int d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        base_input_offset += out_coords[out_coord_idx] * input_strides[d];
        ++out_coord_idx;
    }

    // 沿 dim 维度遍历，记录第一个最小值索引
    int64_t dim_size = input_sizes[dim];
    int64_t dim_stride = input_strides[dim];

    float min_val = input[base_input_offset];
    int64_t min_idx = 0;
    for (int64_t i = 1; i < dim_size; ++i) {
        float val = input[base_input_offset + i * dim_stride];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_offset] = min_idx;
}

// 获取当前 CUDA stream（binding.cpp 调用时传入即可，这里不需要）
// 以下 launcher 供 binding.cpp 声明并调用

extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    const int64_t* sizes,
    const int64_t* strides,
    int ndim,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    if (numel == 0) return;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    clamp_kernel<<<blocks, threads, 0, stream>>>(
        output, input, sizes, strides, ndim, numel, min_val, max_val
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void argmin_forward_launcher(
    const float* input,
    int64_t* output,
    const int64_t* input_sizes,
    const int64_t* input_strides,
    const int64_t* output_sizes,
    const int64_t* output_strides,
    int ndim,
    int dim,
    int64_t output_numel,
    cudaStream_t stream
) {
    if (output_numel == 0) return;
    int threads = 256;
    int blocks = (int)((output_numel + threads - 1) / threads);
    argmin_kernel<<<blocks, threads, 0, stream>>>(
        input, output, input_sizes, input_strides,
        output_sizes, output_strides, ndim, dim, output_numel
    );
    CUDA_CHECK(cudaGetLastError());
}
