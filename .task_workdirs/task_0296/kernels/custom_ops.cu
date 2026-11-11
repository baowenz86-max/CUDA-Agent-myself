#include <cuda_runtime.h>
#include <cstdint>

// 常量内存用于存储 shape 和 strides，最大维度 8
__constant__ int64_t c_shape[8];
__constant__ int64_t c_input_strides[8];
__constant__ int64_t c_output_strides[8];
__constant__ int c_ndim;

__global__ void sign_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < numel; linear += stride) {
        int64_t rem = linear;
        int64_t in_offset = 0;
        int64_t out_offset = 0;
        #pragma unroll
        for (int d = c_ndim - 1; d >= 0; --d) {
            int64_t coord = rem % c_shape[d];
            rem /= c_shape[d];
            in_offset += coord * c_input_strides[d];
            out_offset += coord * c_output_strides[d];
        }

        float val = input[in_offset];
        float result;
        if (isnan(val)) {
            result = val;
        } else if (val > 0.0f) {
            result = 1.0f;
        } else if (val < 0.0f) {
            result = -1.0f;
        } else {
            result = 0.0f;
        }
        output[out_offset] = result;
    }
}

extern "C" void sign_launcher(
    float* output,
    const float* input,
    int64_t numel,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* shape,
    int ndim,
    cudaStream_t stream
) {
    // 将 shape 和 strides 复制到常量内存
    cudaMemcpyToSymbolAsync(c_shape, shape, ndim * sizeof(int64_t), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_input_strides, input_strides, ndim * sizeof(int64_t), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_output_strides, output_strides, ndim * sizeof(int64_t), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_ndim, &ndim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

    const int block_size = 256;
    int64_t num_blocks = (numel + block_size - 1) / block_size;
    if (num_blocks > 2147483647) num_blocks = 2147483647;

    sign_kernel<<<(int)num_blocks, block_size, 0, stream>>>(
        output, input, numel
    );
}
