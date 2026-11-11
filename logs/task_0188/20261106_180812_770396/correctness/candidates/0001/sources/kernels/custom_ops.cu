#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for cumulative minimum along dimension 0 (columns) of a 2D tensor.
// Input: x with strides sx0 (row stride) and sx1 (column stride).
// Output: out with contiguous row-major layout (strides: sy0 = D, sy1 = 1), allocated via empty_like.
// Each thread processes one column sequentially over all rows.
// Launched with enough threads to cover D columns.
__global__ void cummin_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N,          // number of rows
    int D,          // number of columns
    long long sx0,  // input stride for row (in elements)
    long long sx1,  // input stride for column (in elements)
    long long sy0,  // output row stride (should be D)
    long long sy1   // output column stride (should be 1)
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= D) return;

    // Compute base offset for this column (row 0)
    const float* in_ptr = x + col * sx1;
    float* out_ptr = out + col * sy1;

    float running_min = *in_ptr;
    *out_ptr = running_min;

    for (int i = 1; i < N; ++i) {
        float val = *(in_ptr + i * sx0);
        running_min = (val < running_min) ? val : running_min;
        *(out_ptr + i * sy0) = running_min;
    }
}

// Host launcher with C linkage, no PyTorch dependencies.
// Parameters:
//   x: pointer to input float data (CUDA)
//   out: pointer to output float data (CUDA)
//   N: number of rows
//   D: number of columns
//   sx0, sx1: input strides in elements (as int64_t)
//   sy0, sy1: output strides in elements (as int64_t)
//   stream: CUDA stream
extern "C" void cummin_forward_launcher(
    const float* x,
    float* out,
    int N,
    int D,
    int64_t sx0,
    int64_t sx1,
    int64_t sy0,
    int64_t sy1,
    cudaStream_t stream)
{
    // Use a 1D grid of threads covering D columns.
    int threads = 256;
    int blocks = (D + threads - 1) / threads;
    cummin_kernel<<<blocks, threads, 0, stream>>>(
        x, out, N, D, (long long)sx0, (long long)sx1, (long long)sy0, (long long)sy1);
}
