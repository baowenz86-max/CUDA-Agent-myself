#include <cuda_runtime.h>

// CUDA kernel for element-wise max with broadcasting over last two dimensions.
// x: input tensor of shape [batch, channels, height, width] (contiguous)
// tensor: broadcast tensor of shape [height, width] (contiguous)
// total: total number of elements in x
// height, width: dimensions of tensor (and last two dims of x)
__global__ void custom_max_broadcast_kernel(
    const float* __restrict__ x,
    const float* __restrict__ tensor,
    float* __restrict__ out,
    long long total,
    int height,
    int width)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute h,w indices for the last two dimensions.
    int w = idx % width;
    int h = (idx / width) % height;

    float x_val = x[idx];
    float t_val = tensor[(long long)h * width + w];
    out[idx] = (x_val > t_val) ? x_val : t_val;
}

// Launcher function (C interface, no PyTorch dependencies).
// Parameters are raw pointers and dimensions; caller must ensure tensors are CUDA,
// contiguous, and float32. stream is the CUDA stream on which to execute.
extern "C" void custom_max_broadcast_launcher(
    const float* x,
    const float* tensor,
    float* out,
    long long total,
    int height,
    int width,
    cudaStream_t stream)
{
    const int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    custom_max_broadcast_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        x, tensor, out, total, height, width);
}