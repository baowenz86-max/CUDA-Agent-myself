#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Element-wise sigmoid kernel: computes 1 / (1 + exp(-x))
// Supports arbitrary strides via input_stride and output_stride (element-wise per element)
// Grid-stride loop for large tensors.
// -----------------------------------------------------------------------------
__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t total_elements,
                               int64_t input_stride,
                               int64_t output_stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        // Assuming contiguous-ish but stride may be 1 if contiguous; handle general by using stride offset
        // For simplicity, if stride is not 1, we compute per element using base + i*stride.
        // However, in our use case input and output are contiguous, but we keep stride for generality.
        float x = input[i * input_stride];
        float val = 1.0f / (1.0f + expf(-x));
        output[i * output_stride] = val;
    }
}

// -----------------------------------------------------------------------------
// Bias addition kernel: output += bias
// -----------------------------------------------------------------------------
__global__ void bias_add_kernel(float* __restrict__ output,
                                const float* __restrict__ bias,
                                int64_t batch_size,
                                int64_t out_features) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * out_features;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t b = i / out_features;
        int64_t f = i % out_features;
        output[i] += bias[f];
    }
}

// -----------------------------------------------------------------------------
// BatchNorm1d inference kernel: output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
// Optional tensors: if weight is null, use 1; if bias is null, use 0; if running_mean is null, use 0; if running_var is null, use 1.
// Supports arbitrary input/output strides via input_stride and output_stride.
// -----------------------------------------------------------------------------
__global__ void batch_norm1d_inference_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               const float* __restrict__ running_mean,
                                               const float* __restrict__ running_var,
                                               const float* __restrict__ weight,
                                               const float* __restrict__ bias,
                                               float eps,
                                               int64_t total_elements,
                                               int64_t num_features,
                                               int64_t input_stride,
                                               int64_t output_stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t b = i / num_features;
        int64_t f = i % num_features;

        float x = input[i * input_stride];
        float mean = (running_mean != nullptr) ? running_mean[f] : 0.0f;
        float var = (running_var != nullptr) ? running_var[f] : 1.0f;
        float gamma = (weight != nullptr) ? weight[f] : 1.0f;
        float beta = (bias != nullptr) ? bias[f] : 0.0f;

        float inv_std = rsqrtf(var + eps);
        float normalized = (x - mean) * inv_std;
        float result = normalized * gamma + beta;

        output[i * output_stride] = result;
    }
}

// -----------------------------------------------------------------------------
// Launcher: linear_forward
// Computes output = input @ weight.T + bias
// input: [batch, in_features], weight: [out_features, in_features], bias: optional [out_features]
// output: [batch, out_features]
// Uses cuBLAS sgemm with transa=OP_T, transb=OP_N, alpha=1, beta=0
// Then applies bias add if bias is not null (beta=1 would require pre-broadcast, so separate kernel)
// -----------------------------------------------------------------------------
extern "C" void linear_forward_launcher(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        int64_t batch_size,
                                        int64_t in_features,
                                        int64_t out_features,
                                        cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform GEMM: C = A^T * B, where A is weight [out_features x in_features], B is input [in_features x batch]
    // Actually torch linear: input [batch, in] @ weight.T [in, out] -> [batch, out]
    // For cuBLAS column-major format: C (m x n) = A (m x k) * B (k x n)
    // We want C_row_major [batch, out] = input_row_major [batch, in] * weight_row_major^T [out, in]^T
    // In column-major view: C_col_major has shape (out, batch), leading dimension = out_features.
    // A (input) row-major [batch, in] is column-major matrix of shape (in, batch), leading dimension = in_features.
    // B (weight) row-major [out, in] is column-major matrix of shape (in, out), leading dimension = in_features.
    // We need C_col_major = A_col_major^T * B_col_major^T? Instead of transposing, we use cublasSgemm with:
    //   A = input (column-major view shape in x batch, lda=in_features)
    //   B = weight (column-major view shape in x out, ldb=in_features)
    //   Transpose A: A^T = input^T (column-major view of original input? Actually A^T is shape batch x in).
    // Let's compute directly: output_row_major [batch, out] = input_row_major [batch, in] * weight_row_major^T [in, out]
    // In column-major, if we view input_row_major as a column-major matrix with dimensions in x batch, call it A (in x batch).
    // View weight_row_major as column-major matrix with dimensions in x out, call it B (in x out).
    // Then output_row_major [batch, out] corresponds to column-major matrix C with dimensions out x batch.
    // C = B^T * A (out x in) * (in x batch) = out x batch.
    // So we use cublasSgemm with transa=CUBLAS_OP_T, transb=CUBLAS_OP_N:
    //   A = weight (column-major in x out), but we want A as out x in? Actually cublasSgemm computes C = op(A) * op(B).
    // Let A = weight (in x out) as column-major, but we want A^T (out x in). So transa=OP_T.
    // Let B = input (in x batch) as column-major, transb=OP_N.
    // Then C = weight^T * input (out x batch) as column-major, which corresponds to output_row_major [batch, out].
    // Dimensions: m = out_features, n = batch_size, k = in_features.
    // For A (weight) lda = in_features (since column-major matrix of shape in x out has leading dimension in_features).
    // For B (input) ldb = in_features (shape in x batch).
    // For C (column-major out x batch) ldc = out_features.
    // This yields C column-major shape out x batch, stored in our row-major output tensor of shape batch x out.
    // Correction: The above computes C_col = weight^T * input, shape out x batch. In memory, if we have row-major output [batch, out],
    // the column-major C with shape out x batch and leading dimension out_features is exactly the same memory layout.
    // So we set m=out_features, n=batch_size, k=in_features, transa=CUBLAS_OP_T, transb=CUBLAS_OP_N,
    // A=weight, lda=in_features; B=input, ldb=in_features; C=output, ldc=out_features.
    // This is correct and avoids the previous erroneous use of m=batch_size, n=out_features.
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out_features, (int)batch_size, (int)in_features,
                &alpha,
                weight, (int)in_features,
                input, (int)in_features,
                &beta,
                output, (int)out_features);

    cublasDestroy(handle);

    if (bias != nullptr) {
        // Launch bias add kernel
        int block = 256;
        int64_t total = batch_size * out_features;
        int grid = (int)((total + block - 1) / block);
        bias_add_kernel<<<grid, block, 0, stream>>>(output, bias, batch_size, out_features);
    }
}

// -----------------------------------------------------------------------------
// Launcher: sigmoid_forward
// input and output may be non-contiguous, so we use stride parameters.
// The binding layer will compute the strides and pass them.
// -----------------------------------------------------------------------------
extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t total_elements,
                                         int64_t input_stride,
                                         int64_t output_stride,
                                         cudaStream_t stream) {
    int block = 256;
    int max_blocks = 65535;
    int64_t needed_blocks = (total_elements + block - 1) / block;
    int grid = (int)min(needed_blocks, (int64_t)max_blocks);
    if (grid < 1) grid = 1;
    sigmoid_kernel<<<grid, block, 0, stream>>>(input, output, total_elements, input_stride, output_stride);
}

// -----------------------------------------------------------------------------
// Launcher: batch_norm1d_forward
// input: [batch, num_features], output same.
// running_mean, running_var, weight, bias are optional (nullptr if not provided).
// Strides handled via input_stride/output_stride.
// -----------------------------------------------------------------------------
extern "C" void batch_norm1d_forward_launcher(const float* input,
                                             float* output,
                                             const float* running_mean,
                                             const float* running_var,
                                             const float* weight,
                                             const float* bias,
                                             double eps,
                                             int64_t total_elements,
                                             int64_t num_features,
                                             int64_t input_stride,
                                             int64_t output_stride,
                                             cudaStream_t stream) {
    int block = 256;
    int max_blocks = 65535;
    int64_t needed_blocks = (total_elements + block - 1) / block;
    int grid = (int)min(needed_blocks, (int64_t)max_blocks);
    if (grid < 1) grid = 1;
    batch_norm1d_inference_kernel<<<grid, block, 0, stream>>>(
        input, output, running_mean, running_var, weight, bias,
        (float)eps, total_elements, num_features, input_stride, output_stride);
}
