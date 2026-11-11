#include <cuda_runtime.h>

// Kernel with general stride support
__global__ void cumprod_along_dim0_kernel_general(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim0,
    int dim1,
    int dim2,
    long long input_stride0,
    long long input_stride1,
    long long input_stride2,
    long long output_stride0,
    long long output_stride1,
    long long output_stride2)
{
    int inner_size = dim1 * dim2;
    int inner_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (inner_idx >= inner_size) return;

    // Compute base offsets for this inner index using modular arithmetic.
    // inner_idx = idx1 * dim2 + idx2 (assuming row-major order)
    int idx2 = inner_idx % dim2;
    int idx1 = (inner_idx / dim2) % dim1;

    long long input_base = idx1 * input_stride1 + idx2 * input_stride2;
    long long output_base = idx1 * output_stride1 + idx2 * output_stride2;

    float cum = 1.0f;
    for (int i = 0; i < dim0; ++i) {
        cum *= input[input_base + i * input_stride0];
        output[output_base + i * output_stride0] = cum;
    }
}

// Launcher: computes cumprod along dim0 for a 3D tensor with arbitrary strides.
extern "C" void cumprod_launcher(
    const float* input,
    float* output,
    int dim0,
    int dim1,
    int dim2,
    long long input_stride0,
    long long input_stride1,
    long long input_stride2,
    long long output_stride0,
    long long output_stride1,
    long long output_stride2,
    cudaStream_t stream)
{
    int inner_size = dim1 * dim2;
    int block_size = 256;
    int grid_size = (inner_size + block_size - 1) / block_size;

    cumprod_along_dim0_kernel_general<<<grid_size, block_size, 0, stream>>>(
        input, output, dim0, dim1, dim2,
        input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, output_stride2);
}
