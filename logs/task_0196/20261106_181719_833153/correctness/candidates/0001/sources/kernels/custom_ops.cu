#include <cuda_runtime.h>
#include <cmath>

// GELU activation (exact erf-based, matching PyTorch F.gelu default)
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long numel,
                            int dim0, int dim1, int dim2,
                            long long stride0, long long stride1, long long stride2) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (; idx < numel; idx += stride) {
        long long i0 = idx / ((long long)dim1 * dim2);
        long long rem = idx - i0 * ((long long)dim1 * dim2);
        long long i1 = rem / dim2;
        long long i2 = rem - i1 * dim2;
        long long offset = i0 * stride0 + i1 * stride1 + i2 * stride2;
        float x = input[offset];
        float y = 0.5f * x * (1.0f + erff(x * inv_sqrt2));
        output[offset] = y;
    }
}

// Tanh activation
template <int BLOCK_SIZE>
__global__ void tanh_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long numel,
                            int dim0, int dim1, int dim2,
                            long long stride0, long long stride1, long long stride2) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        long long i0 = idx / ((long long)dim1 * dim2);
        long long rem = idx - i0 * ((long long)dim1 * dim2);
        long long i1 = rem / dim2;
        long long i2 = rem - i1 * dim2;
        long long offset = i0 * stride0 + i1 * stride1 + i2 * stride2;
        float x = input[offset];
        output[offset] = tanhf(x);
    }
}

// Launcher: GELU
extern "C" void gelu_forward_launcher(const float* input,
                                     float* output,
                                     long long numel,
                                     int dim0, int dim1, int dim2,
                                     long long stride0, long long stride1, long long stride2,
                                     cudaStream_t stream) {
    const int block_size = 256;
    // Calculate grid size: number of blocks needed to cover numel with one element per thread initially
    long long needed = (numel + block_size - 1) / block_size;
    // Cap grid to max grid size (avoid exceeding 2^31-1)
    int max_blocks = 2147483647;
    int grid = (int)min(needed, (long long)max_blocks);
    gelu_kernel<<<grid, block_size, 0, stream>>>(input, output, numel, dim0, dim1, dim2, stride0, stride1, stride2);
}

// Launcher: Tanh
extern "C" void tanh_forward_launcher(const float* input,
                                     float* output,
                                     long long numel,
                                     int dim0, int dim1, int dim2,
                                     long long stride0, long long stride1, long long stride2,
                                     cudaStream_t stream) {
    const int block_size = 256;
    long long needed = (numel + block_size - 1) / block_size;
    int max_blocks = 2147483647;
    int grid = (int)min(needed, (long long)max_blocks);
    tanh_kernel<256><<<grid, block_size, 0, stream>>>(input, output, numel, dim0, dim1, dim2, stride0, stride1, stride2);
}
