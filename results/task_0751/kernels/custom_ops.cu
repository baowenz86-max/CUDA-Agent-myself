#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// CUDA kernels for model_new.py
//
// This file contains all CUDA kernels and host launchers.
// No torch/ATen/c10/pybind dependencies.
//
// Operators implemented:
//   1. linear_forward: y = input @ weight^T + bias
//   2. minimum_forward: elementwise minimum with broadcasting along last dim
// -----------------------------------------------------------------------------

// Kernel: one thread per output element (row, col) computes dot product of
// input[row, :] and weight[col, :] plus bias[col].
__global__ void linear_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    // Grid-stride loop over all (row, col) pairs
    int total = batch_size * out_features;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        int row = idx / out_features;
        int col = idx % out_features;

        const float* input_row = input + (size_t)row * in_features;
        const float* weight_col = weight + (size_t)col * in_features;

        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input_row[k] * weight_col[k];
        }
        sum += bias[col];

        output[(size_t)row * out_features + col] = sum;
    }
}

// Kernel: elementwise minimum, broadcasting other (length out_features)
// along the last dimension.
__global__ void minimum_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ other,
    float* __restrict__ output,
    int64_t numel,
    int64_t out_features)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        int64_t col = i % out_features;
        float a = input[i];
        float b = other[col];
        output[i] = (a < b) ? a : b;  // matches torch.minimum (NaN handling not required)
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C" so they can be called from binding.cpp)
// -----------------------------------------------------------------------------

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    cudaStream_t stream)
{
    dim3 block(256);
    // Cap grid size and rely on grid-stride loop
    int max_grid = 65535;  // conservative for all architectures
    int64_t total = batch_size * out_features;
    int grid = (int)std::min((int64_t)max_grid, (total + block.x - 1) / block.x);
    if (grid < 1) grid = 1;

    linear_forward_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        (int)batch_size, (int)in_features, (int)out_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw here; binding layer may check via cudaDeviceSynchronize
        // We do nothing; user must handle as appropriate.
    }
}

extern "C" void minimum_forward_launcher(
    const float* input,
    const float* other,
    float* output,
    int64_t numel,
    int64_t out_features,
    cudaStream_t stream)
{
    dim3 block(256);
    int64_t total = numel;
    int max_grid = 65535;
    int grid = (int)std::min((int64_t)max_grid, (total + block.x - 1) / block.x);
    if (grid < 1) grid = 1;

    minimum_forward_kernel<<<grid, block, 0, stream>>>(
        input, other, output, numel, out_features);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // silent failure; binding layer can check with cudaDeviceSynchronize
    }
}
