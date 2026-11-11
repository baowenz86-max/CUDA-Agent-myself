#include <cuda_runtime.h>

// Sum over dimension 1 (channels) with keepdim=True.
// Input: x of shape [N, C, H, W] contiguous float32.
// Output: y of shape [N, 1, H, W] contiguous float32.
// Each thread handles one output element (n,0,h,w), looping over C.
__global__ void sum_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int64_t total_elements = (int64_t)N * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int n = i / ((int64_t)H * W);
        int64_t rem = i - n * (int64_t)H * W;
        int h = rem / W;
        int w = rem - h * W;

        float sum = 0.0f;
        const float* x_base = x + ((int64_t)n * C) * H * W;
        for (int c = 0; c < C; ++c) {
            sum += x_base[((int64_t)c * H + h) * W + w];
        }
        y[i] = sum;
    }
}

// Replace NaN with zero, keep other values.
__global__ void nan_to_num_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t numel)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float v = x[i];
        y[i] = isnan(v) ? 0.0f : v;
    }
}

// Launcher for custom_sum_dim1_keepdim
extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int64_t total_elements = (int64_t)N * H * W;
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    // Cap grid to avoid exceeding max grid dimension (2^31-1).
    const int max_blocks = 1048576; // 2^20, safe for all GPUs
    if (blocks > max_blocks) blocks = max_blocks;

    sum_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W);
}

// Launcher for custom_nan_to_num
extern "C" void nan_to_num_launcher(
    const float* x,
    float* y,
    int64_t numel,
    cudaStream_t stream)
{
    int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    // Cap blocks to avoid overflow and excessive grid size.
    const int64_t max_blocks = 1048576;
    int blocks = (int)(blocks64 < max_blocks ? blocks64 : max_blocks);

    nan_to_num_kernel<<<blocks, threads, 0, stream>>>(
        x, y, numel);
}
