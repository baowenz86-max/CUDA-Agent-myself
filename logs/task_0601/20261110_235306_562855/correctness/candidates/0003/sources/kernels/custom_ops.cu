#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err));                                    \
            return;                                                             \
        }                                                                       \
    } while (0)

// Kernel for concatenation along dimension dim, specialized for rank 4.
// Assumes inputs are contiguous or have standard strides; for simplicity,
// we require contiguous inputs (the reference torch.cat on CUDA likely uses contiguous inputs).
// The binding will make inputs contiguous before calling.
__global__ void concat_kernel_4d(
    float* __restrict__ out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    long long total_elems,
    long long B,
    long long C_out,
    long long H,
    long long W,
    long long C_split
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long linear = idx; linear < total_elems; linear += stride) {
        // Decompose linear index: linear = (((b * C_out + c) * H + h) * W + w)
        long long w = linear % W;
        long long tmp = linear / W;
        long long h = tmp % H;
        tmp /= H;
        long long c = tmp % C_out;
        long long b = tmp / C_out;

        const float* src;
        long long src_linear;
        if (c < C_split) {
            src = x1;
            src_linear = ((b * C_split + c) * H + h) * W + w;
        } else {
            src = x2;
            long long c2 = c - C_split;
            src_linear = ((b * C_split + c2) * H + h) * W + w;
        }
        out[linear] = src[src_linear];
    }
}

// Host launcher for concat_dim (rank 4, contiguous inputs)
extern "C" void concat_dim_launcher(
    float* out,
    const float* x1,
    const float* x2,
    long long total_elems,
    long long B,
    long long C_split,
    long long H,
    long long W,
    cudaStream_t stream
) {
    long long C_out = 2 * C_split;
    int block_size = 256;
    long long grid_size = (total_elems + block_size - 1) / block_size;
    if (grid_size > 2147483647) grid_size = 2147483647;
    concat_kernel_4d<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        out, x1, x2, total_elems, B, C_out, H, W, C_split);
    CUDA_CHECK(cudaGetLastError());
}
