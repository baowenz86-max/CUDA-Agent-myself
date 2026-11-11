#include <cuda_runtime.h>

// ============================================================================
// custom_mean_dim1: computes mean over channel dimension (dim=1) with keepdim.
// Input: x (float*) with shape [N, C, H, W] and strides (batch_stride, channel_stride, height_stride, width_stride).
// Output: out (float*) with shape [N, 1, H, W] (contiguous, N*H*W elements).
// Each thread computes one output element: sum over C, divide by C.
// ============================================================================
__global__ void mean_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    long long batch_stride, long long channel_stride,
    long long height_stride, long long width_stride,
    long long total_out) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total_out; idx += stride) {
        // Decompose idx into n, h, w (output has shape N x 1 x H x W)
        long long n = idx / ((long long)H * W);
        long long rem = idx - n * ((long long)H * W);
        long long h = rem / W;
        long long w = rem - h * W;

        float sum = 0.0f;
        const float* base = x + n * batch_stride + h * height_stride + w * width_stride;
        for (int c = 0; c < C; ++c) {
            sum += base[c * channel_stride];
        }
        out[idx] = sum / (float)C;
    }
}

// Launcher for custom_mean_dim1. Returns cudaError_t; caller should check.
extern "C" cudaError_t custom_mean_dim1_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long batch_stride, long long channel_stride,
    long long height_stride, long long width_stride,
    cudaStream_t stream) {

    long long total_out = (long long)N * H * W;
    if (total_out == 0) {
        return cudaSuccess;
    }

    const int threads = 256;
    long long blocks_ll = (total_out + threads - 1) / threads;
    // Cap grid dimension to 2^31-1; kernel has grid-stride loop so this is safe.
    int blocks = (int)min(blocks_ll, (long long)2147483647);

    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, out, N, C, H, W,
        batch_stride, channel_stride, height_stride, width_stride,
        total_out);

    return cudaGetLastError();
}

// ============================================================================
// custom_leaky_relu: elementwise LeakyReLU y = x if x > 0 else slope*x.
// Input: x contiguous float* with 'numel' elements.
// Output: out contiguous float* with same numel elements.
// ============================================================================
__global__ void leaky_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long numel,
    float negative_slope) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < numel; idx += stride) {
        float v = x[idx];
        out[idx] = (v > 0.0f) ? v : (negative_slope * v);
    }
}

// Launcher for custom_leaky_relu. Returns cudaError_t; caller should check.
extern "C" cudaError_t custom_leaky_relu_launcher(
    const float* x,
    float* out,
    long long numel,
    double negative_slope,
    cudaStream_t stream) {

    if (numel == 0) {
        return cudaSuccess;
    }

    const int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    int blocks = (int)min(blocks_ll, (long long)2147483647);

    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(
        x, out, numel, (float)negative_slope);

    return cudaGetLastError();
}
