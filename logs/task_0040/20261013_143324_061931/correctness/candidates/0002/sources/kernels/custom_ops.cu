// CUDA kernels and launchers for non-convolution custom ops.
// Conv/ConvTranspose forward is implemented in a separate C++ file using cuDNN.
// This file contains only raw CUDA kernels and launchers.

#include <cuda_runtime.h>
#include <cublas_v2.h>

// -----------------------------------------------------------------------------
// channel_add: x = x + broadcast bias (bias shape [C])
// x: [N, C, L] float32, bias: [C] float32
// Each thread processes one element: output[n,c,l] = x[n,c,l] + bias[c]
// -----------------------------------------------------------------------------
__global__ void channel_add_kernel(const float* __restrict__ x,
                                   const float* __restrict__ bias,
                                   float* __restrict__ y,
                                   int64_t N, int64_t C, int64_t L) {
    int64_t total = N * C * L;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute n, c, l from flattened index
    int64_t l = idx % L;
    int64_t tmp = idx / L;
    int64_t c = tmp % C;
    // n = tmp / C; not needed

    y[idx] = x[idx] + bias[c];
}

extern "C" void channel_add_launcher(const float* x,
                                     const float* bias,
                                     float* y,
                                     int64_t N, int64_t C, int64_t L,
                                     cudaStream_t stream) {
    int64_t total = N * C * L;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 0) {
        channel_add_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, N, C, L);
    }
}

// -----------------------------------------------------------------------------
// roll_last_dim: circular shift along last dimension by shift (positive means right shift)
// x: [N, C, L], shift: integer, output separate (cannot be in-place)
// For each output position i in last dim, new_val = x[(i - shift + L) % L]
// -----------------------------------------------------------------------------
__global__ void roll_last_dim_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int64_t N, int64_t C, int64_t L,
                                     int64_t shift) {
    int64_t total = N * C * L;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t l = idx % L;
    int64_t base = idx - l; // offset to start of this (n,c) row
    int64_t src_l = (l - shift) % L;
    if (src_l < 0) src_l += L;
    y[idx] = x[base + src_l];
}

extern "C" void roll_last_dim_launcher(const float* x,
                                       float* y,
                                       int64_t N, int64_t C, int64_t L,
                                       int64_t shift,
                                       cudaStream_t stream) {
    int64_t total = N * C * L;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 0) {
        roll_last_dim_kernel<<<blocks, threads, 0, stream>>>(x, y, N, C, L, shift);
    }
}

// -----------------------------------------------------------------------------
// flatten_to_2d: no data movement, just returns a view in Python.
// No kernel needed.
// We provide a dummy launcher that does nothing (binding can ignore it).
// -----------------------------------------------------------------------------
extern "C" void flatten_to_2d_launcher(cudaStream_t stream) {
    // Nothing to do; the binding layer will return x.view({N, C*L}).
    (void)stream;
}

// -----------------------------------------------------------------------------
// linear_forward: y = x @ W^T + bias
// x: [N, in_features], W: [out_features, in_features], bias: [out_features]
// y: [N, out_features]
// Uses cuBLAS for the GEMM, then a custom kernel for bias addition.
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* __restrict__ y,
                                const float* __restrict__ bias,
                                int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t col = idx % cols;
    y[idx] += bias[col];
}

extern "C" void linear_forward_launcher(const float* x,
                                        const float* weight,
                                        const float* bias, // may be nullptr
                                        float* y,
                                        int64_t N, int64_t in_features, int64_t out_features,
                                        cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute y = x @ weight^T
    // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
    // We want y (N x out_features) = x (N x in_features) * weight^T (in_features x out_features)
    // In cuBLAS column-major, we treat A = weight (in_features x out_features) stored in row-major?
    // Actually PyTorch stores weight as [out_features, in_features] row-major.
    // To perform C = A * B where A is N x in_features (row-major) and B is out_features x in_features (row-major),
    // we need to transpose B. Using cublasSgemm with:
    //   opA = CUBLAS_OP_N (A is N x in_features, but cuBLAS sees column-major so effectively A^T?)
    // Better approach: use cublasSgemm with opA=CUBLAS_OP_T, opB=CUBLAS_OP_N as follows:
    //   M = out_features, N = N, K = in_features
    //   A = weight (out_features x in_features) but we treat as (in_features x out_features) via transpose.
    //   B = x (N x in_features) but we treat as (in_features x N) via transpose.
    // Actually standard PyTorch linear uses cublasSgemm with:
    //   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, N, in_features,
    //               &alpha, weight, in_features, x, in_features, &beta, y, out_features);
    // This computes y = weight^T * x^T? Wait, that would result in out_features x N.
    // Let's derive: cublas is column-major, but PyTorch tensors are row-major,
    // so passing row-major pointers is like passing transposed matrices to cuBLAS.
    // The common recipe for row-major C = A * B is:
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
    //               &alpha, B, n, A, k, &beta, C, n);
    // where A is m x k row-major, B is k x n row-major.
    // Here we have A = x (N x in_features), B = weight^T (in_features x out_features).
    // So m = N, k = in_features, n = out_features.
    // We need to compute C (N x out_features) = x @ weight^T.
    // Using the row-major recipe with opA=CUBLAS_OP_N, opB=CUBLAS_OP_N:
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, N, in_features,
    //               &alpha, weight, out_features, x, in_features, &beta, y, out_features);
    // Wait, that would treat weight as column-major (out_features x in_features) but we want transpose.
    // Let's use the known working approach from PyTorch: for linear with weight [out_features, in_features],
    // they call cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, N, in_features,
    //                      &alpha, weight, in_features, x, in_features, &beta, y, out_features);
    // That is because weight is stored as [out_features, in_features] row-major, but cublas sees it as
    // column-major with dimensions (in_features, out_features), so opA=CUBLAS_OP_T makes it effectively
    // (out_features, in_features) column-major. Similarly x is [N, in_features] row-major, cublas sees it as
    // column-major (in_features, N), so opB=CUBLAS_OP_N keeps it as (in_features, N).
    // The multiplication yields (out_features, N) and then we get y as column-major, but we want row-major
    // N x out_features. However, the resulting y pointer is interpreted as column-major, but PyTorch treats it as row-major,
    // so the resulting layout is actually transposed. This is the standard trick: it works because we then store it in row-major,
    // effectively giving the transpose. So we use that exact call.
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, N, in_features,
                &alpha,
                weight, in_features,
                x, in_features,
                &beta,
                y, out_features);

    // Add bias if provided
    if (bias != nullptr) {
        int threads = 256;
        int64_t total = N * out_features;
        int64_t blocks = (total + threads - 1) / threads;
        if (blocks > 0) {
            add_bias_kernel<<<blocks, threads, 0, stream>>>(y, bias, N, out_features);
        }
    }

    cublasDestroy(handle);
}
