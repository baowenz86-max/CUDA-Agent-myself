#include <cuda_runtime.h>

// Elementwise hardtanh clamp: out = min(max(x, min_val), max_val)
__global__ void hardtanh_clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel,
    float min_val,
    float max_val) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = input[idx];
        v = fminf(fmaxf(v, min_val), max_val);
        output[idx] = v;
    }
}

// Mean over channel dimension (dim=1) with keepdim=True.
// Input assumed to be 4D contiguous with sizes [N,C,H,W] in row-major order.
// Output is [N,1,H,W], also contiguous.
// Each thread computes one output spatial position (h,w) for a batch n,
// summing over channels.
__global__ void mean_channel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {
    // Compute linear index for output element (n, 0, h, w)
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_output = N * H * W;
    if (idx >= total_output) return;

    // Decompose idx into n, h, w
    int64_t n = idx / (H * W);
    int64_t hw = idx % (H * W);
    int64_t h = hw / W;
    int64_t w = hw % W;

    // Base input index for (n, 0, h, w)
    int64_t input_base = (n * C) * (H * W) + h * W + w;

    float sum = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        sum += input[input_base + c * (H * W)];
    }
    output[idx] = sum / static_cast<float>(C);
}

// Launcher for hardtanh_clamp. Input and output have same shape and are contiguous.
// The binding layer is responsible for passing contiguous tensors; we assume that.
extern "C" void hardtanh_clamp_launcher(
    const float* input,
    float* output,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // grid-size limit: use loop if needed, but numel max ~805M, grid up to ~3.1M within limit
    hardtanh_clamp_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, numel, min_val, max_val);
}

// Launcher for mean_channel reduction. Input contiguous [N,C,H,W]. Output contiguous [N,1,H,W].
extern "C" void mean_channel_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream) {
    int64_t total_output = N * H * W;
    if (total_output <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (total_output + block_size - 1) / block_size;
    mean_channel_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W);
}
