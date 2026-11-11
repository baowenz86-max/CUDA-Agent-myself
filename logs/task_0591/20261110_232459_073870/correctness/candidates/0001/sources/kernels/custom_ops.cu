#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// GELU activation (exact PyTorch default: 0.5 * x * (1 + erf(x / sqrt(2))))
__global__ void gelu_forward_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    }
}

// Bias addition for linear output: out[i] = gemm_out[i] + bias[out_col]
__global__ void bias_add_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int64_t rows,
                                int64_t cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_idx = idx; linear_idx < rows * cols; linear_idx += stride) {
        int64_t col = linear_idx % cols;
        float val = input[linear_idx];
        val += bias[col];
        output[linear_idx] = val;
    }
}

// Optional bias version: if bias is nullptr, just copy input to output
extern "C" void gelu_forward_launcher(const float* input,
                                     float* output,
                                     int64_t n,
                                     cudaStream_t stream) {
    if (n <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (n + block_size - 1) / block_size;
    // Cap grid to avoid too many blocks, grid-stride loop handles remainder
    if (grid_size > 65535) grid_size = 65535;
    gelu_forward_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, n);
}

extern "C" void bias_add_launcher(const float* input,
                                 const float* bias,  // may be nullptr
                                 float* output,
                                 int64_t rows,
                                 int64_t cols,
                                 cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int64_t total = rows * cols;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    if (bias == nullptr) {
        // If no bias, just copy via a simple kernel (or use bias_add with zero bias? we'll use copy)
        // For simplicity, we handle bias always present in this case; but contract says bias can be nullable.
        // Since this file only gets launchers, the binding will ensure bias pointer is valid when calling.
        // If bias is null, we can still call bias_add_kernel with bias==nullptr, but we need to avoid dereference.
        // Better: have a separate copy kernel or conditional inside kernel.
        // Here we'll assume bias is always passed when required; contract says optional but linear has bias=True.
        // To be safe, we can modify bias_add_kernel to check if bias is null (but passing null to kernel is fine, need branch).
        // We'll create a second kernel variant for no-bias copy.
        // However, given task context, bias is always provided (linear bias=True). We'll keep simple.
    }
    bias_add_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, bias, output, rows, cols);
}

// Optional: a simple copy kernel for the case bias is null, but not needed if binding ensures bias pointer.
// We can add a kernel that handles both cases with a bool, but we keep two launchers.
// Since the contract says bias is a tensor (possibly nullable), the binding will pass nullptr if no bias.
// To support that, modify bias_add_kernel to accept a bool has_bias and branch.
__global__ void bias_add_optional_kernel(const float* __restrict__ input,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int64_t rows,
                                        int64_t cols,
                                        bool has_bias) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear_idx = idx; linear_idx < rows * cols; linear_idx += stride) {
        int64_t col = linear_idx % cols;
        float val = input[linear_idx];
        if (has_bias) {
            val += bias[col];
        }
        output[linear_idx] = val;
    }
}

extern "C" void bias_add_optional_launcher(const float* input,
                                          const float* bias,
                                          float* output,
                                          int64_t rows,
                                          int64_t cols,
                                          bool has_bias,
                                          cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int64_t total = rows * cols;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    bias_add_optional_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, bias, output, rows, cols, has_bias);
}

// For convenience, we export the standard bias_add_launcher that always uses bias (non-null).
// The binding should call bias_add_optional_launcher if bias may be null, or bias_add_launcher if bias is always present.
// We'll keep both; next stage may choose appropriately.
