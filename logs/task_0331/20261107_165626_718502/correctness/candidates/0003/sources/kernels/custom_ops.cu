// custom_ops.cu - CUDA kernels and launchers for ModelNew
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

// -----------------------------------------------------------------------------
// Linear forward: y = x @ weight.T + bias
// x: [batch, in_features], weight: [in_features, in_features], bias: [in_features]
// Output y: [batch, in_features]
// -----------------------------------------------------------------------------
__global__ void linear_forward_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ y,
                                      int batch, int in_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = batch * in_features;
    for (int i = idx; i < total; i += stride) {
        int row = i / in_features;
        int col = i % in_features;
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += x[row * in_features + k] * weight[col * in_features + k]; // fixed transposition
        }
        y[i] = sum + bias[col];
    }
}

// -----------------------------------------------------------------------------
// GELU forward (default approximate='none'): y = x * Phi(x)
// Use precise formula matching PyTorch CUDA implementation
// -----------------------------------------------------------------------------
__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = x[i];
        // Match PyTorch's GELU (approximate='none'): v * 0.5 * (1 + erf(v / sqrt(2)))
        // Use double constant for high precision, then cast to float
        float gelu_val = (float)(0.5 * v * (1.0 + erff(v * 0.70710678118654752440)));
        y[i] = gelu_val;
    }
}

// -----------------------------------------------------------------------------
// Greater-than scalar: y = (x > threshold) as bool
// -----------------------------------------------------------------------------
__global__ void gt_scalar_forward_kernel(const float* __restrict__ x,
                                         float threshold,
                                         bool* __restrict__ y,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = (x[i] > threshold);
    }
}

// -----------------------------------------------------------------------------
// Bool add bias: x bool [batch, in_features], bias float [in_features,1,1]
// Output: float [in_features, batch, in_features]
// -----------------------------------------------------------------------------
__global__ void bool_add_bias_forward_kernel(const bool* __restrict__ x,
                                             const float* __restrict__ bias,
                                             float* __restrict__ out,
                                             int batch, int in_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = in_features * batch * in_features;
    int batch_times_in = batch * in_features;
    for (int i = idx; i < total; i += stride) {
        int b = i / batch_times_in;
        int rem = i % batch_times_in;
        int r = rem / in_features;
        int c = rem % in_features;
        float val = (x[r * in_features + c] ? 1.0f : 0.0f) + bias[b];
        out[i] = val;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// -----------------------------------------------------------------------------
extern "C" void linear_forward_launcher(const float* x,
                                         const float* weight,
                                         const float* bias,
                                         float* y,
                                         int batch,
                                         int in_features,
                                         cudaStream_t stream) {
    int total = batch * in_features;
    int block = 256;
    int grid = (total + block - 1) / block;
    const int max_grid = 1024;
    if (grid > max_grid) grid = max_grid;
    linear_forward_kernel<<<grid, block, 0, stream>>>(x, weight, bias, y, batch, in_features);
}

extern "C" void gelu_forward_launcher(const float* x,
                                       float* y,
                                       int n,
                                       cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    const int max_grid = 1024;
    if (grid > max_grid) grid = max_grid;
    gelu_forward_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

extern "C" void gt_scalar_forward_launcher(const float* x,
                                            float threshold,
                                            bool* y,
                                            int n,
                                            cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    const int max_grid = 1024;
    if (grid > max_grid) grid = max_grid;
    gt_scalar_forward_kernel<<<grid, block, 0, stream>>>(x, threshold, y, n);
}

extern "C" void bool_add_bias_forward_launcher(const bool* x,
                                               const float* bias,
                                               float* out,
                                               int batch,
                                               int in_features,
                                               cudaStream_t stream) {
    int total = in_features * batch * in_features;
    int block = 256;
    int grid = (total + block - 1) / block;
    const int max_grid = 1024;
    if (grid > max_grid) grid = max_grid;
    bool_add_bias_forward_kernel<<<grid, block, 0, stream>>>(x, bias, out, batch, in_features);
}
