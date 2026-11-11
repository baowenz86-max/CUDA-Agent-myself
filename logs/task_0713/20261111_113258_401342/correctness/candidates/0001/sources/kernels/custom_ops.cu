#include <cuda_runtime.h>
#include <cmath>

// Kernel: compute mean over dimension 1 (channels) with keepdim=True
// Input: x [B, C, H, W] contiguous float32
// Output: out [B, 1, H, W]
__global__ void mean_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B,
    int C,
    int H,
    int W)
{
    int64_t total = (int64_t)B * H * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < total; idx += stride) {
        int b = idx / ((int64_t)H * W);
        int64_t rem = idx - (int64_t)b * H * W;
        int h = rem / W;
        int w = rem - (int64_t)h * W;

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            int64_t in_idx = ((int64_t)b * C + c) * H * W + (int64_t)h * W + w;
            sum += x[in_idx];
        }
        out[idx] = sum / C;
    }
}

// Kernel: elementwise cosine
__global__ void cos_elementwise_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t numel)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < numel; i += stride) {
        out[i] = cosf(x[i]);
    }
}

// Launcher for mean_dim1_keepdim
extern "C" void mean_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream)
{
    int64_t total = (int64_t)B * H * W;
    const int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    // grid cap for very large total (avoid exceeding int max)
    const int max_blocks = 65535;
    int blocks = (int)std::min<int64_t>(blocks64, max_blocks);

    mean_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In typical binding, this would be reported; here simply ignore
        // but avoid compiler warning
        (void)err;
    }
}

// Launcher for cos_elementwise
extern "C" void cos_elementwise_launcher(
    const float* x,
    float* out,
    int64_t numel,
    cudaStream_t stream)
{
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    const int max_blocks = 65535;
    int blocks = (int)std::min<int64_t>(blocks64, max_blocks);

    cos_elementwise_kernel<<<blocks, threads, 0, stream>>>(
        x, out, numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        (void)err;
    }
}
