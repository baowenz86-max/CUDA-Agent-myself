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
    // For cuBLAS, we need dimensions: m = batch, n = out_features, k = in_features
    // C (m x n) = A (m x k) * B (k x n)
    // A = input (batch x in), B = weight.T? But weight.T is [in, out], which is B (k x n). So we need B = weight.T (not stored).
    // Alternative: use cublasSgemm with transa=CUBLAS_OP_T, transb=CUBLAS_OP_N: C = A^T * B where A is weight (out x in), B is input (in x batch)
    // Then C would be out x batch, not batch x out. We need batch x out, so we can compute C^T = A * B^T where A is input (batch x in), B is weight (out x in).
    // Use cublasSgemm with transa=CUBLAS_OP_N, transb=CUBLAS_OP_T on A=input, B=weight: C = A * B^T = input @ weight.T, dimensions: m=batch, n=out_features, k=in_features.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                (int)batch_size, (int)out_features, (int)in_features,
                &alpha,
                input, (int)in_features,   // A: batch x in (leading dimension = in_features)
                weight, (int)in_features,   // B: out x in (leading dimension = in_features), but with trans, effectively in x out
                &beta,
                output, (int)batch_size);

    cublasDestroy(handle);

    if (bias != nullptr) {
        // Launch bias add kernel
        int block = 256;
        int grid = (int)((batch_size * out_features + block - 1) / block);
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
    // For typical contiguous tensors, stride is 1. But we support arbitrary.
    int block = 256;
    // Grid size limited to avoid launching too many blocks; use grid-stride loop safely
    int max_blocks = 65535;
    int64_t needed_blocks = (total_elements + block - 1) / block;
    int grid = (int)min(needed_blocks, (int64_t)max_blocks);
    // For small tensors, needed_blocks may be smaller
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
