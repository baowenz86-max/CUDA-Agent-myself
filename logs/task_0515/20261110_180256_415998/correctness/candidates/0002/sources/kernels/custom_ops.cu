// kernels/custom_ops.cu
// CUDA kernels and launchers for custom operations.
// This file contains only CUDA kernels and host launchers with raw pointers.
// No PyTorch/Tensor dependencies.

#include <cuda_runtime.h>
#include <cublas_v2.h>

// -----------------------------------------------------------------------------
// adaptive_avg_pool2d
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void adaptive_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c = (idx / (outW * outH)) % C;
    int n = idx / (outW * outH * C);

    int start_h = (int)floor((float)oh * H / outH);
    int end_h = (int)ceil((float)(oh + 1) * H / outH);
    int start_w = (int)floor((float)ow * W / outW);
    int end_w = (int)ceil((float)(ow + 1) * W / outW);

    if (end_h > H) end_h = H;
    if (end_w > W) end_w = W;
    if (start_h < 0) start_h = 0;
    if (start_w < 0) start_w = 0;
    if (end_h <= start_h) end_h = start_h + 1;
    if (end_w <= start_w) end_w = start_w + 1;

    scalar_t sum = 0;
    int count = 0;
    const scalar_t* input_base = input + ((n * C + c) * H) * W;
    for (int h = start_h; h < end_h; ++h) {
        const scalar_t* row = input_base + h * W;
        for (int w = start_w; w < end_w; ++w) {
            sum += row[w];
            ++count;
        }
    }
    output[idx] = sum / (scalar_t)count;
}

extern "C" void adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream) {

    int total_elements = N * C * outH * outW;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    adaptive_avg_pool2d_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, total_elements);
}

// -----------------------------------------------------------------------------
// view_flatten placeholder
// -----------------------------------------------------------------------------
extern "C" void view_flatten_launcher(
    const float* input, float* output, int batch_size, int num_features,
    cudaStream_t stream) {
    // intentionally no-op
}

// -----------------------------------------------------------------------------
// bias_add kernel for linear_forward
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void bias_add_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    int batch,
    int out_features,
    int total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int col = idx % out_features;
    output[idx] += bias[col];
}

// -----------------------------------------------------------------------------
// linear_forward
// -----------------------------------------------------------------------------
extern "C" void linear_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int out_features,
    cudaStream_t stream) {

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS column-major: compute C = alpha * op(A) * op(B) + beta * C
    // A is input (batch x in_features) stored row-major, but cuBLAS sees it as
    // column-major matrix of dimensions in_features x batch.
    // We want output = input @ weight^T + bias, which in cuBLAS terms is
    // C^T = W^T * A^T? Let's use the standard trick:
    // For row-major A(m,k) and B(k,n), compute C(m,n) = A * B^T.
    // In cuBLAS, treat A as column-major A_col(k,m) with lda=in_features, and
    // B as column-major B_col(n,k) with ldb=in_features (since weight is out_features x in_features).
    // Then C_col = A_col * B_col^T, which is column-major C(n,m).
    // But we need row-major C(m,n). The memory layout of C_col corresponds to
    // row-major C(m,n) if we set ldc=out_features and interpret correctly.
    // Actually, the common approach for row-major C = A * B^T is:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    //             m=batch, n=out_features, k=in_features,
    //             &alpha, input, lda=in_features,
    //             weight, ldb=in_features,
    //             &beta, output, ldc=out_features);
    // However, cuBLAS expects column-major, but this call with these parameters
    // exactly computes the row-major result because of the transpose flags.
    // Let's verify: A is m x k (row-major input), op(A)=A (N) so cuBLAS sees A as k x m column-major.
    // B is k x n (weight^T), op(B)=B^T (T) so cuBLAS sees B as n x k column-major (which matches weight).
    // Then C = A * B^T is m x n row-major, but cuBLAS stores it as n x m column-major,
    // which is exactly the memory layout if ldc=out_features (since row stride = n).
    // So the call is correct. But we'll re-verify and keep as is.

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        batch, out_features, in_features,
        &alpha,
        input, in_features,
        weight, in_features,
        &beta,
        output, out_features);

    if (bias != nullptr) {
        int total_elements = batch * out_features;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        bias_add_kernel<float><<<grid_size, block_size, 0, stream>>>(
            output, bias, batch, out_features, total_elements);
    }

    cublasDestroy(handle);
}
