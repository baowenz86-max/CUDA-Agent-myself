#include <cuda_runtime.h>

// Elementwise softsign kernel: out = x / (1 + |x|)
// Supports arbitrary contiguous/non-contiguous strides via 4D index calculation.
// Grid-stride loop handles any numel without grid dimension limits.
__global__ void softsign_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t rem = linear;
        int64_t i3 = rem % dim3;
        rem /= dim3;
        int64_t i2 = rem % dim2;
        rem /= dim2;
        int64_t i1 = rem % dim1;
        rem /= dim1;
        int64_t i0 = rem;

        int64_t offset = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
        float val = input[offset];
        output[offset] = val / (1.0f + fabsf(val));
    }
}

// Launcher: raw pointers, shape/stride as int64_t, current CUDA stream.
// No PyTorch/Tensor dependencies. Caller must ensure memory is valid and stream is current.
extern "C" void softsign_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    cudaStream_t stream)
{
    if (total_elements == 0) {
        return; // nothing to compute
    }

    const int threads = 256;
    int64_t max_blocks_per_grid = 2147483647; // 2^31-1, device limit for gridDim.x
    int64_t blocks = (total_elements + threads - 1) / threads;
    if (blocks > max_blocks_per_grid) {
        blocks = max_blocks_per_grid;
    }

    dim3 grid((unsigned int)blocks);
    dim3 block(threads);

    softsign_kernel<<<grid, block, 0, stream>>>(
        input, output,
        total_elements,
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3);
}
