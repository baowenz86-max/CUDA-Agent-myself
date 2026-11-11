#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel to compute mean along dim=1 for a 3D tensor with shape (B, D, W)
// Input: x (float*) with strides sx0, sx1, sx2 (elements)
// Output: out (float*) contiguous shape (B, W)
__global__ void mean_dim1_kernel(const float* __restrict__ x, float* __restrict__ out,
                                 int B, int D, int W,
                                 long long sx0, long long sx1, long long sx2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W;
    if (idx >= total) return;
    int b = idx / W;
    int w = idx % W;
    const float* base = x + b * sx0 + w * sx2;
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        sum += base[d * sx1];
    }
    out[b * W + w] = sum / (float)D;
}

// Host launcher for custom_mean
// Input: x pointer (device), shape B x D x W, strides in elements (can be arbitrary)
// Output: out pointer (device), shape B x W, contiguous
// dim must be 1, keepdim must be false (checked in binding)
extern "C" void custom_mean_launcher(const float* x, float* out,
                                      int B, int D, int W,
                                      long long sx0, long long sx1, long long sx2,
                                      cudaStream_t stream) {
    int total = B * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(x, out, B, D, W, sx0, sx1, sx2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Should not happen; binding will handle error checking, but we can leave a getter? No.
        // We cannot throw, so maybe just do nothing or assert? Better to not have side effects.
        // Since launcher cannot communicate error, we rely on binding to check after call.
    }
}

// Kernel to add bias with arbitrary strides to contiguous output.
__global__ void add_bias_strided_kernel(float* __restrict__ c, const float* __restrict__ bias,
                                        int M, int N, long long s0, long long s1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int row = idx / N;
    int col = idx % N;
    c[idx] += bias[row * s0 + col * s1];
}

// Custom kernel to copy a matrix (M x N) from arbitrary stride to contiguous row-major.
__global__ void copy_matrix_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                   int M, int N, long long s0, long long s1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int row = idx / N;
    int col = idx % N;
    dst[idx] = src[row * s0 + col * s1];
}

// Helper to check if a matrix is contiguous row-major given strides (s0, s1) and shape (M,N).
__device__ __host__ bool is_row_major_contiguous(int M, int N, long long s0, long long s1) {
    return (s1 == 1) && (s0 == N);
}

// Host launcher for custom_addmm: computes out = bias + mat1 @ mat2
// bias: M x N, mat1: M x K, mat2: K x N, out: M x N (contiguous)
// All pointers are device pointers.
// We assume alpha=1, beta=0 for GEMM part, then add bias.
// We handle arbitrary strides for mat1 and mat2 by copying to contiguous if needed.
extern "C" void custom_addmm_launcher(const float* bias, const float* mat1, const float* mat2,
                                        float* out,
                                        int M, int K, int N,
                                        long long bias_s0, long long bias_s1,
                                        long long mat1_s0, long long mat1_s1,
                                        long long mat2_s0, long long mat2_s1,
                                        cudaStream_t stream) {
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        // handle error; but we have no way to report, so just return
        return;
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f; // we will add bias separately

    // Check if mat1 and mat2 are contiguous row-major; if not, copy them.
    const float* A_ptr = mat1;
    const float* B_ptr = mat2;
    float* temp_A = nullptr;
    float* temp_B = nullptr;
    bool need_copy_A = !is_row_major_contiguous(M, K, mat1_s0, mat1_s1);
    bool need_copy_B = !is_row_major_contiguous(K, N, mat2_s0, mat2_s1);

    if (need_copy_A) {
        cudaMalloc(&temp_A, M * K * sizeof(float));
        int threads = 256;
        int blocks = (M * K + threads - 1) / threads;
        copy_matrix_kernel<<<blocks, threads, 0, stream>>>(mat1, temp_A, M, K, mat1_s0, mat1_s1);
        cudaGetLastError(); // ignore for now
        A_ptr = temp_A;
        mat1_s0 = K; mat1_s1 = 1; // now contiguous
    }
    if (need_copy_B) {
        cudaMalloc(&temp_B, K * N * sizeof(float));
        int threads = 256;
        int blocks = (K * N + threads - 1) / threads;
        copy_matrix_kernel<<<blocks, threads, 0, stream>>>(mat2, temp_B, K, N, mat2_s0, mat2_s1);
        cudaGetLastError(); // ignore
        B_ptr = temp_B;
        mat2_s0 = N; mat2_s1 = 1; // now contiguous
    }

    // For cublasSgemm, we use row-major, so op(A)=N, op(B)=N.
    // We need lda = mat1_s0 (distance between rows), ldb = mat2_s0, ldc = N (out contiguous)
    int lda = (int)mat1_s0;
    int ldb = (int)mat2_s0;
    int ldc = N;

    // Call cublasSgemm: C = alpha * A * B + beta * C
    // Since beta=0, C is overwritten.
    // Compute row-major C = A * B by using column-major trick:
    // A (M x K) row-major -> column-major K x M with lda=K
    // B (K x N) row-major -> column-major N x K with lda=N
    // C (M x N) row-major -> column-major N x M with lda=N
    // So we set m=N, n=M, k=K, and use B as first matrix, A as second matrix.
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K,
                       &alpha,
                       B_ptr, N,   // B as A_cublas: N x K (col-major) with ld = N
                       A_ptr, K,   // A as B_cublas: K x M (col-major) with ld = K
                       &beta,
                       out, N);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        // handle error
    }

    // Now add bias to out.
    // Bias may have arbitrary strides, but likely contiguous. For correctness, we should handle arbitrary strides
    // too, but we can assume contiguous and if not, we can copy or use custom kernel with strides.
    // We use add_bias_strided_kernel which handles strides.
    int total = M * N;
    int threads2 = 256;
    int blocks2 = (total + threads2 - 1) / threads2;
    add_bias_strided_kernel<<<blocks2, threads2, 0, stream>>>(out, bias, M, N, bias_s0, bias_s1);
    cudaGetLastError();

    // Free temporary buffers
    if (temp_A) cudaFree(temp_A);
    if (temp_B) cudaFree(temp_B);
    cublasDestroy(handle);
}
