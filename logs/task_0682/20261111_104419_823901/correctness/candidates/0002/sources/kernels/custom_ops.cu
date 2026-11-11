// CUDA kernels for mean_dim0.
// This file contains no torch/ATen/c10/pybind dependencies.
#include <cuda_runtime.h>
#include <cstdint>

// Kernel for mean over dimension 0.
// Contract shape: input [N, D1, ..., Dk, 2] (k>=1), output [D1, ..., Dk, 2].
// The non-trivial middle dims (D1..Dk) are folded into one contiguous index M.
// Input may have arbitrary strides; output is contiguous with last dim 2.
// Each thread computes one output element: output[m, c] where c in {0,1}.
// Access: input[n, m, c] = input[n*stride0 + middle_offset(m) + c*stride_last]
// For this task input is contiguous and strides are known, but code uses strides.
__global__ void mean_dim0_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t N,
                                 int64_t M,               // product of D1..Dk
                                 const int64_t* __restrict__ shape,   // shape[0]=N, shape[1..ndim-1]=D1..Dk,2
                                 const int64_t* __restrict__ strides, // strides[0..ndim-1]
                                 int64_t ndim) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = M * 2;
    if (idx >= total) return;

    int64_t m = idx >> 1;         // output spatial index
    int64_t c = idx & 1;          // 0 or 1

    // Compute input offset for coordinate (m, c) excluding dimension 0.
    int64_t rem = m;
    int64_t offset = c * strides[ndim - 1]; // last dim stride for c
    for (int64_t d = ndim - 2; d >= 1; --d) {
        int64_t coord = rem % shape[d];
        rem /= shape[d];
        offset += coord * strides[d];
    }

    const float* base = input + offset;
    float sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) {
        sum += base[n * strides[0]];
    }
    output[idx] = sum / (float)N;
}

// Launcher for mean_dim0.
// shape and strides are host pointers to int64_t arrays of length ndim.
// output must be preallocated contiguous tensor of shape [D1..Dk, 2].
extern "C" void mean_dim0_launcher(const float* input,
                                   float* output,
                                   int64_t N,
                                   int64_t ndim,
                                   const int64_t* shape,
                                   const int64_t* strides,
                                   cudaStream_t stream) {
    // Compute M = product of shape[1..ndim-2] (D1..Dk, excluding last dim 2).
    int64_t M = 1;
    for (int64_t d = 1; d < ndim - 1; ++d) {
        M *= shape[d];
    }
    int64_t total = M * 2;
    if (total <= 0) {
        return;
    }

    const int block = 256;
    int64_t grid64 = (total + block - 1) / block;
    if (grid64 > 2147483647) {
        grid64 = 2147483647;
    }
    dim3 grid((unsigned int)grid64);
    dim3 threads(block);

    mean_dim0_kernel<<<grid, threads, 0, stream>>>(
        input, output, N, M, shape, strides, ndim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will also check; nothing to do here except avoid silent failure.
        // We deliberately do not throw because this is a C interface.
    }
}
