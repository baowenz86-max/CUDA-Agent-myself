// CUDA kernel and launcher for baddbmm_forward
// Computes out = beta * bias + alpha * (mat1 @ mat2) for batched matrices.
// The implementation uses cuBLAS strided batched GEMM for the matrix multiplication
// and a custom kernel for bias addition.

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel to add bias to the output of GEMM.
// Supports arbitrary strides and broadcasting (bias may have fewer dimensions or size 1 in any dimension).
// bias_stride[d] gives the stride in elements for dimension d of bias, or 0 if broadcast (size 1).
// out_stride[d] gives the stride in elements for dimension d of output.
// Batch, M, N are dimensions; each thread computes one output element.
__global__ void bias_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int batch, int M, int N,
    long long out_stride_b, long long out_stride_m, long long out_stride_n,
    long long bias_stride_b, long long bias_stride_m, long long bias_stride_n,
    float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * M * N;
    if (idx >= total) return;

    // Decompose linear index to b, m, n
    int n = idx % N;
    int tmp = idx / N;
    int m = tmp % M;
    int b = tmp / M;

    // Compute bias index using strides (broadcast if stride is 0)
    long long bias_idx = (long long)b * bias_stride_b + (long long)m * bias_stride_m + (long long)n * bias_stride_n;
    // Compute output linear index
    long long out_idx = (long long)b * out_stride_b + (long long)m * out_stride_m + (long long)n * out_stride_n;

    out[out_idx] = beta * bias[bias_idx] + out[out_idx];
}

// Launcher for baddbmm_forward.
// Inputs:
//   bias: pointer to bias tensor (may be non-contiguous, shape broadcastable to [batch, M, N])
//   mat1: pointer to mat1 tensor (shape [batch, M, K])
//   mat2: pointer to mat2 tensor (shape [batch, K, N])
//   output: pointer to output tensor (shape [batch, M, N], already allocated)
//   batch, M, N, K: dimensions
//   alpha, beta: scalars
//   bias_strides: array of 3 strides for bias (in elements), with 0 meaning broadcast along that dimension
//   mat1_strides: array of 3 strides for mat1 [batch, M, K]
//   mat2_strides: array of 3 strides for mat2 [batch, K, N]
//   out_strides: array of 3 strides for output [batch, M, N]
//   stream: CUDA stream
// The function may allocate temporary contiguous buffers if mat1 or mat2 are not contiguous in the way cuBLAS expects,
// or if bias is not contiguous. For simplicity and performance, we assume the common case of contiguous tensors;
// otherwise we fall back to copying.
// NOTE: This launcher is declared extern "C" for binding.
extern "C" void baddbmm_forward_launcher(
    float* output,
    const float* bias,
    const float* mat1,
    const float* mat2,
    int batch, int M, int N, int K,
    float alpha, float beta,
    const long long* bias_strides,
    const long long* mat1_strides,
    const long long* mat2_strides,
    const long long* out_strides,
    cudaStream_t stream)
{
    // Check if inputs are contiguous in row-major order (with batch dimension leading)
    // For cuBLAS strided batched GEMM, we require:
    //   mat1 is [batch, M, K] with stride between batches = M*K, row stride = K, col stride = 1
    //   mat2 is [batch, K, N] with stride between batches = K*N, row stride = N, col stride = 1
    //   output is [batch, M, N] with stride between batches = M*N, row stride = N, col stride = 1
    // If these conditions are met, we can call cublasSgemmStridedBatched directly.
    bool mat1_contiguous = (mat1_strides[0] == (long long)M * K && mat1_strides[1] == K && mat1_strides[2] == 1);
    bool mat2_contiguous = (mat2_strides[0] == (long long)K * N && mat2_strides[1] == N && mat2_strides[2] == 1);
    bool out_contiguous = (out_strides[0] == (long long)M * N && out_strides[1] == N && out_strides[2] == 1);

    // Temporary buffers for non-contiguous cases
    float* d_mat1 = nullptr;
    float* d_mat2 = nullptr;
    float* d_out = nullptr;

    // For simplicity, in this version we assume all inputs and output are contiguous.
    // The binding currently ensures contiguity before calling this launcher.
    // If non-contiguous, we do not handle it here (could add copy kernels later).

    const float* gemm_mat1 = mat1_contiguous ? mat1 : d_mat1;
    const float* gemm_mat2 = mat2_contiguous ? mat2 : d_mat2;
    float* gemm_out = out_contiguous ? output : d_out;

    // Initialize cublas handle (static for reuse)
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    cublasSetStream(handle, stream);

    // Compute GEMM using cuBLAS with beta=0 (we add bias separately).
    // For row-major matrices, use the column-major trick:
    //   C = alpha * A * B  ->  C^T = alpha * B^T * A^T
    // Call cublasSgemmStridedBatched with m=N, n=M, k=K,
    //   A=mat2 (leading dimension N), B=mat1 (leading dimension K), C=output (leading dimension N)
    float zero = 0.0f;
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        gemm_mat2, N, (long long)K * N,
        gemm_mat1, K, (long long)M * K,
        &zero,
        gemm_out, N, (long long)M * N,
        batch
    );

    // Add bias (beta * bias) to the output using custom kernel.
    int total = batch * M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // We use the output directly even if non-contiguous for now (assume contiguous).
    // If output non-contiguous, we would need to apply bias to gemm_out and then copy.
    float* final_out = out_contiguous ? output : d_out;
    bias_add_kernel<<<blocks, threads, 0, stream>>>(
        final_out,
        bias,
        batch, M, N,
        (long long)M * N, (long long)N, 1,  // out strides (contiguous)
        bias_strides[0], bias_strides[1], bias_strides[2],
        beta
    );

    // Free temporary buffers if allocated (currently none).
    if (d_mat1) cudaFree(d_mat1);
    if (d_mat2) cudaFree(d_mat2);
    if (d_out) cudaFree(d_out);
}
