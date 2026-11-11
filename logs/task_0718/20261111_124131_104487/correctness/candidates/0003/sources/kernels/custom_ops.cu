#include <cuda_runtime.h>
#include <cstdint>

// custom_sum_dim1: reduce tensor x of shape [B, C, H, W] along dimension 1.
// Output shape [B, H, W]. Assumes contiguous input and output.
__global__ void custom_sum_dim1_kernel(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W;
    if (idx >= total) return;

    int b = idx / (H * W);
    int rem = idx - b * (H * W);
    int h = rem / W;
    int w = rem - h * W;

    float sum = 0.0f;
    const float* base = x + (b * C * H * W) + h * W + w;
    for (int c = 0; c < C; ++c) {
        sum += base[c * H * W];
    }
    out[idx] = sum;
}

// custom_fmod: elementwise fmod of input tensor of shape [N].
__global__ void custom_fmod_kernel(
    const float* x,
    float* out,
    int n,
    double divisor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = fmodf(x[idx], (float)divisor);
}

// Launcher for custom_sum_dim1
// x: input tensor [B,C,H,W] contiguous
// out: output tensor [B,H,W] contiguous
// stream: CUDA stream
extern "C" void custom_sum_dim1_launcher(
    const float* x,
    float* out,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream)
{
    int total = (int)(B * H * W);
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    custom_sum_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, (int)B, (int)C, (int)H, (int)W);
}

// Launcher for custom_fmod
// x: input tensor [N] contiguous
// out: output tensor [N] contiguous
// divisor: scalar divisor (double)
// stream: CUDA stream
extern "C" void custom_fmod_launcher(
    const float* x,
    float* out,
    int64_t n,
    double divisor,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (int)((n + block_size - 1) / block_size);
    custom_fmod_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, (int)n, divisor);
}
