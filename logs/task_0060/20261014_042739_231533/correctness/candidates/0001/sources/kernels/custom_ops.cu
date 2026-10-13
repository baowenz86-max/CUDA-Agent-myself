// kernels/custom_ops.cu
#include <cuda_runtime.h>

__global__ void fused_relu_hardsigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    int shape_ndim,
    const long long* __restrict__ input_strides,
    const long long* __restrict__ output_strides)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < numel; idx += stride) {
        // compute multi-dimensional indices from linear idx
        long long linear = idx;
        long long in_offset = 0;
        long long out_offset = 0;
        for (int d = shape_ndim - 1; d >= 0; --d) {
            long long coord = linear % shape[d];
            linear /= shape[d];
            in_offset += coord * input_strides[d];
            out_offset += coord * output_strides[d];
        }
        float val = input[in_offset];
        val = val > 0.0f ? val : 0.0f;  // relu
        val = val + 3.0f;
        val = val < 0.0f ? 0.0f : (val > 6.0f ? 6.0f : val); // clamp
        output[out_offset] = val * (1.0f / 6.0f);
    }
}

extern "C" void fused_relu_hardsigmoid_launcher(
    float* output,
    const float* input,
    long long numel,
    int shape_ndim,
    const long long* input_strides,
    const long long* output_strides,
    cudaStream_t stream)
{
    // Choose block size and grid size
    int block_size = 256;
    // Grid-stride loop to cover numel with a reasonable number of blocks
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    fused_relu_hardsigmoid_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, numel, shape_ndim, input_strides, output_strides);
}
