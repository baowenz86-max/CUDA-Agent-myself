// CUDA kernels and launchers for custom ops
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>
#include <iostream>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

// -----------------------------------------------------------------------------
// ConvTranspose1d kernel
// Computes output[b, c_out, l_out] = bias[c_out] + sum_{c_in, k} input[b, c_in, l_in] * weight[c_in, c_out, k]
// where l_out = l_in * stride - padding + k, and l_out must satisfy 0 <= l_out < L_out
// L_out = (L_in-1)*stride - 2*padding + K + output_padding
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,   // [B, C_in, L_in]
    const float* __restrict__ weight,  // [C_in, C_out, K]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ output,        // [B, C_out, L_out]
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int output_padding, int L_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C_out * L_out) return;

    int l_out = idx % L_out;
    int tmp = idx / L_out;
    int c_out = tmp % C_out;
    int b = tmp / C_out;

    float sum = 0.0f;
    if (bias != nullptr) {
        sum = bias[c_out];
    }

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            // Solve for l_in: l_out = l_in * stride - padding + k
            // => l_in * stride = l_out - k + padding
            int numerator = l_out - k + padding;
            if (numerator < 0) continue;
            if (numerator % stride != 0) continue;
            int l_in = numerator / stride;
            if (l_in < 0 || l_in >= L_in) continue;
            sum += input[(b * C_in + c_in) * L_in + l_in] * weight[(c_in * C_out + c_out) * K + k];
        }
    }
    output[idx] = sum;
}

// -----------------------------------------------------------------------------
// Bias add kernel: y[b, c, i] = x[b, c, i] + bias[c]
__global__ void bias_add_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int L,
    long long in_stride_b, long long in_stride_c, long long in_stride_l,
    long long out_stride_b, long long out_stride_c, long long out_stride_l)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * C * L;
    if (idx >= total) return;
    int l = idx % L;
    long long tmp = idx / L;
    int c = tmp % C;
    int b = tmp / C;
    output[b * out_stride_b + c * out_stride_c + l * out_stride_l] =
        input[b * in_stride_b + c * in_stride_c + l * in_stride_l] + bias[c];
}

// -----------------------------------------------------------------------------
// Roll kernel: shifts=1, dims=2 (last dim) cyclic shift.
__global__ void roll_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int L,
    long long in_stride_b, long long in_stride_c, long long in_stride_l,
    long long out_stride_b, long long out_stride_c, long long out_stride_l)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * C * L;
    if (idx >= total) return;
    int l = idx % L;
    long long tmp = idx / L;
    int c = tmp % C;
    int b = tmp / C;
    int src_l = (l - 1 + L) % L;
    output[b * out_stride_b + c * out_stride_c + l * out_stride_l] =
        input[b * in_stride_b + c * in_stride_c + src_l * in_stride_l];
}

// -----------------------------------------------------------------------------
// Matrix bias add kernel for linear layer (adds bias to each row)
__global__ void add_bias_matrix_kernel(float* out, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int col = idx % N;
    out[idx] += bias[col];
}

// -----------------------------------------------------------------------------
// Launcher for conv_transpose1d (pure CUDA kernel, no cuDNN)
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int L_in, int C_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    int total = B * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        B, C_in, L_in, C_out, K,
        stride, padding, output_padding, L_out);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Launcher for bias_add_forward
extern "C" void bias_add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int L,
    long long in_stride_b, long long in_stride_c, long long in_stride_l,
    long long out_stride_b, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream)
{
    long long total = (long long)B * C * L;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    bias_add_kernel<<<blocks, threads, 0, stream>>>(
        input, bias, output, B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Launcher for roll_forward
extern "C" void roll_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    long long in_stride_b, long long in_stride_c, long long in_stride_l,
    long long out_stride_b, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream)
{
    long long total = (long long)B * C * L;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    roll_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Launcher for linear_forward using cuBLAS
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int M, int K, int N,
    cudaStream_t stream)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            M, N, K,
                            &alpha,
                            input, K,
                            weight, K,
                            &beta,
                            output, N));

    if (bias != nullptr) {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        add_bias_matrix_kernel<<<blocks, threads, 0, stream>>>(output, bias, M, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUBLAS_CHECK(cublasDestroy(handle));
}
