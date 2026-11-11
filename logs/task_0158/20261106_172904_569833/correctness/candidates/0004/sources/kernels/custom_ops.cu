#include <cuda_runtime.h>

// Kernel to compute mean over channel dimension (C=3)
// Input: x with shape [B, C, H, W] and strides (stride_b, stride_c, stride_h, stride_w)
// Output: y with shape [B, H, W] contiguous
__global__ void reduce_mean_channel_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long total_elements,  // B * H * W
    int C,
    long long stride_b,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        // Decompose idx into (b, h, w) assuming output contiguous [B, H, W]
        long long w = idx % out_stride_w;
        long long tmp = idx / out_stride_w;
        long long h = tmp % out_stride_h;
        long long b = tmp / out_stride_h;

        const float* base = x + (long long)b * stride_b + (long long)h * stride_h + w * stride_w;
        float sum = 0.0f;
        // Unroll for C=3 (known constant) for performance
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            sum += base[c * stride_c];
        }
        y[idx] = sum / (float)C;
    }
}

// Kernel to add scalar to every element
__global__ void add_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long total_elements,
    float scalar)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        y[idx] = x[idx] + scalar;
    }
}

// Launcher for reduce_mean_channel
// Assumes input is float32 CUDA tensor, output allocated as contiguous [B,H,W]
// Input may be non-contiguous; strides provided.
// Returns cudaError_t for easy error checking in binding.
extern "C" cudaError_t launch_reduce_mean_channel(
    const float* x,
    float* y,
    long long B,
    long long C,
    long long H,
    long long W,
    long long stride_b,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream)
{
    long long total_elements = B * H * W;
    if (total_elements <= 0) {
        return cudaSuccess; // nothing to do
    }

    const int block_size = 256;
    // Calculate grid size (capped to avoid excessive blocks, grid-stride loop handles rest)
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 1024) grid_size = 1024; // common max grid limit for 1D

    reduce_mean_channel_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, total_elements, (int)C,
        stride_b, stride_c, stride_h, stride_w,
        H * W, W, 1  // output contiguous strides: out_stride_b=H*W, out_stride_h=W, out_stride_w=1
    );

    return cudaGetLastError();
}

// Launcher for add_scalar
// Input and output are contiguous float32 tensors with same shape
extern "C" cudaError_t launch_add_scalar(
    const float* x,
    float* y,
    long long total_elements,
    float scalar,
    cudaStream_t stream)
{
    if (total_elements <= 0) {
        return cudaSuccess;
    }

    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 1024) grid_size = 1024;

    add_scalar_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, total_elements, scalar);

    return cudaGetLastError();
}
