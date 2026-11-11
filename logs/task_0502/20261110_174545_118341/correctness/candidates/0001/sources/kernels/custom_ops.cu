// kernels/custom_ops.cu
// CUDA kernels and launchers for matmul (cuBLAS) and softmax along dim=0.
// No PyTorch/ATen headers, only raw pointers and cudaStream_t.

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Utility macro for CUDA error checking (used only in host launchers).
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            // Do not throw; the binding layer may check cudaGetLastError later. \
            return;                                                             \
        }                                                                       \
    } while (0)

//-----------------------------------------------------------------------------
// Matmul launcher using cuBLAS SGEMM.
// Inputs are float32, row-major (as in PyTorch default) with dimensions:
//   x: [M, K], y: [K, N], output: [M, N]
// The launcher assumes all pointers are valid CUDA device pointers.
// If inputs are not contiguous, the binding layer should pass contiguous copies.
// The launcher creates and destroys a cuBLAS handle for each call;
// for repeated calls it could be cached, but this keeps the .cu self-contained.
//-----------------------------------------------------------------------------
extern "C" void matmul_launcher(const float* x, const float* y, float* out,
                                int M, int N, int K, cudaStream_t stream) {
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        return;
    }
    stat = cublasSetStream(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major storage; for row-major we compute C^T = B^T A^T.
    // Here we want C = A * B, where A is MxK, B is KxN, C is MxN.
    // Column-major: C^T = B^T * A^T, dimensions: N x M = (N x K) * (K x M).
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, y, N, x, K, &beta, out, N)
    // Let's verify: y is KxN in row-major, but in column-major y^T is NxK with leading dim N.
    // x is MxK row-major, x^T is KxM column-major with leading dim K.
    // out is MxN row-major, out^T is NxM column-major with leading dim N.
    // The above should produce out^T = y^T * x^T, so out is correct.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                y, N,
                x, K,
                &beta,
                out, N);

    cublasDestroy(handle);
}

//-----------------------------------------------------------------------------
// Softmax along dim=0 kernel.
// Each block processes one column (N = 2048 columns).
// Input/output are contiguous, shape [rows, cols] = [2048, 2048].
// We use a two-pass approach:
//   1. Find max per column (numerical stability).
//   2. Compute exp(x - max) and sum.
//   3. Normalize.
// Shared memory: blockDim.x threads, need reduction for max and sum.
// We'll use warp-level reductions for speed.
//-----------------------------------------------------------------------------
__global__ void softmax_dim0_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int rows, int cols) {
    // One block per column (gridDim.x = cols).
    int col = blockIdx.x;
    if (col >= cols) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Stride loop for rows if rows > block_size.
    // rows = 2048, typical block_size = 256, so 8 iterations.
    // Compute max.
    float max_val = -CUDART_INF_F;
    for (int row = tid; row < rows; row += block_size) {
        float val = input[row * cols + col];
        max_val = fmaxf(max_val, val);
    }

    // Reduce max across block (all threads).
    // Use shared memory for block reduction.
    __shared__ float s_max[256]; // blockDim.x must be <= 256
    s_max[tid] = max_val;
    __syncthreads();

    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }
    max_val = s_max[0];

    // Compute sum of exp(x - max).
    float sum_exp = 0.0f;
    for (int row = tid; row < rows; row += block_size) {
        float val = __expf(input[row * cols + col] - max_val);
        // Store exp in shared memory? Not necessary, we recompute later.
        sum_exp += val;
    }

    // Reduce sum across block.
    __shared__ float s_sum[256];
    s_sum[tid] = sum_exp;
    __syncthreads();

    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    sum_exp = s_sum[0];

    // Normalize and write output.
    float inv_sum = 1.0f / sum_exp;
    for (int row = tid; row < rows; row += block_size) {
        float val = __expf(input[row * cols + col] - max_val) * inv_sum;
        output[row * cols + col] = val;
    }
}

// Launcher for softmax_dim0: each column processed by one block.
// gridDim.x = cols, blockDim.x = 256 (fixed).
extern "C" void softmax_dim0_launcher(const float* input, float* output,
                                      int rows, int cols, cudaStream_t stream) {
    const int block_size = 256;
    dim3 grid(cols);
    dim3 block(block_size);
    softmax_dim0_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
}
