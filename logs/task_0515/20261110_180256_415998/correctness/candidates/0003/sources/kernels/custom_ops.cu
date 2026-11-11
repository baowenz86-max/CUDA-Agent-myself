// kernels/custom_ops.cu
// CUDA kernels and launchers for custom operations.
// This file contains only CUDA kernels and host launchers with raw pointers.
// No PyTorch/Tensor dependencies.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// adaptive_avg_pool2d
// -----------------------------------------------------------------------------
// Computes adaptive average pooling for 4D input (N, C, H, W)
// with output size (outH, outW).
// Each output element at (n, c, oh, ow) averages over the window
// [start_h, end_h) x [start_w, end_w) where
// start_h = floor(oh * H / outH), end_h = ceil((oh+1) * H / outH)
// similarly for width.

template <typename scalar_t>
__global__ void adaptive_avg_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Index decomposition: output is contiguous [N, C, outH, outW]
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

// Launcher for float32 (only dtype used in contract)
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
// view_flatten
// -----------------------------------------------------------------------------
// Since view returns a view, it must be done in binding.cpp with Tensor::view.
// This .cu cannot implement it without Tensor operations.
// We provide a dummy launcher declaration that binding can ignore or not use.
// However, for completeness, we could implement a copy-based flatten if desired,
// but contract says "No copy is performed; returns a view".
// So we do not implement any CUDA kernel for view_flatten. The binding layer
// will directly return x.view({batch_size, -1}) without calling CUDA.
// To satisfy the requirement that each operator has a launcher, we define a
// no-op launcher that does nothing, but it should not be called by binding.
extern "C" void view_flatten_launcher(
    const float* input, float* output, int batch_size, int num_features,
    cudaStream_t stream) {
    // This is intentionally a no-op placeholder; actual view is done in C++.
    // It will not be used in the binding since binding directly returns a view.
}

// -----------------------------------------------------------------------------
// linear_forward (custom CUDA kernel, no cuBLAS)
// -----------------------------------------------------------------------------
// Computes output = input @ weight^T + bias
// input: [batch, in_features] (row-major)
// weight: [out_features, in_features] (row-major)
// bias: [out_features] (may be null)
// output: [batch, out_features] (row-major)
// Each thread computes one output element.

template <typename scalar_t>
__global__ void linear_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_features,
    int out_features) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;
    if (idx >= total) return;

    int b = idx / out_features;
    int o = idx % out_features;

    scalar_t sum = 0;
    for (int k = 0; k < in_features; ++k) {
        sum += input[b * in_features + k] * weight[o * in_features + k];
    }
    if (bias != nullptr) {
        sum += bias[o];
    }
    output[idx] = sum;
}

// Launcher for float32
extern "C" void linear_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int out_features,
    cudaStream_t stream) {

    int total_elements = batch * out_features;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    linear_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch, in_features, out_features);
}
