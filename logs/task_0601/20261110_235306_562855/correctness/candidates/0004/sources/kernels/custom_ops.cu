// CUDA kernels for concat_dim and rot90_k1_dims23
// Specialized for 4D float32 tensors as required by the task.
// Concat copies x1 then x2 along the specified dimension into a contiguous output.
// Rot90 is handled as a view in the binding (no kernel needed).

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

// 4D concat kernel
__global__ void concat_kernel_4d(
    float* __restrict__ out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    long long total_elems,
    int dim,
    long long out_s0, long long out_s1, long long out_s2, long long out_s3,
    long long x1_s0, long long x1_s1, long long x1_s2, long long x1_s3,
    long long x2_s0, long long x2_s1, long long x2_s2, long long x2_s3,
    long long c1  // size of x1 along dim
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long linear = idx; linear < total_elems; linear += stride) {
        // Decompose linear index into 4D output indices (contiguous output)
        long long rem = linear;
        long long i0 = rem / out_s0; rem %= out_s0;
        long long i1 = rem / out_s1; rem %= out_s1;
        long long i2 = rem / out_s2; rem %= out_s2;
        long long i3 = rem; // out_s3 should be 1 for contiguous

        // Determine source tensor and corresponding index along dim
        long long src_linear;
        if (dim == 0) {
            if (i0 < c1) {
                src_linear = i0 * x1_s0 + i1 * x1_s1 + i2 * x1_s2 + i3 * x1_s3;
                out[linear] = x1[src_linear];
            } else {
                long long i0_adj = i0 - c1;
                src_linear = i0_adj * x2_s0 + i1 * x2_s1 + i2 * x2_s2 + i3 * x2_s3;
                out[linear] = x2[src_linear];
            }
        } else if (dim == 1) {
            if (i1 < c1) {
                src_linear = i0 * x1_s0 + i1 * x1_s1 + i2 * x1_s2 + i3 * x1_s3;
                out[linear] = x1[src_linear];
            } else {
                long long i1_adj = i1 - c1;
                src_linear = i0 * x2_s0 + i1_adj * x2_s1 + i2 * x2_s2 + i3 * x2_s3;
                out[linear] = x2[src_linear];
            }
        } else if (dim == 2) {
            if (i2 < c1) {
                src_linear = i0 * x1_s0 + i1 * x1_s1 + i2 * x1_s2 + i3 * x1_s3;
                out[linear] = x1[src_linear];
            } else {
                long long i2_adj = i2 - c1;
                src_linear = i0 * x2_s0 + i1 * x2_s1 + i2_adj * x2_s2 + i3 * x2_s3;
                out[linear] = x2[src_linear];
            }
        } else { // dim == 3
            if (i3 < c1) {
                src_linear = i0 * x1_s0 + i1 * x1_s1 + i2 * x1_s2 + i3 * x1_s3;
                out[linear] = x1[src_linear];
            } else {
                long long i3_adj = i3 - c1;
                src_linear = i0 * x2_s0 + i1 * x2_s1 + i2 * x2_s2 + i3_adj * x2_s3;
                out[linear] = x2[src_linear];
            }
        }
    }
}

extern "C" void concat_dim_launcher(
    float* out,
    const float* x1,
    const float* x2,
    long long total_elems,
    int dim,
    long long out_s0, long long out_s1, long long out_s2, long long out_s3,
    long long x1_s0, long long x1_s1, long long x1_s2, long long x1_s3,
    long long x2_s0, long long x2_s1, long long x2_s2, long long x2_s3,
    long long c1,
    cudaStream_t stream
) {
    int block_size = 256;
    long long grid_size = (total_elems + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        grid_size = 2147483647;
    }
    concat_kernel_4d<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        out, x1, x2, total_elems, dim,
        out_s0, out_s1, out_s2, out_s3,
        x1_s0, x1_s1, x1_s2, x1_s3,
        x2_s0, x2_s1, x2_s2, x2_s3,
        c1
    );
    CUDA_CHECK(cudaGetLastError());
}
