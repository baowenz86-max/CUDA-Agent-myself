#include <cuda_runtime.h>
#include <math.h>

// Element-wise sigmoid: y = 1 / (1 + exp(-x))
// Supports arbitrary strides via scalar arguments.
__global__ void custom_sigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int linear = idx; linear < total_elements; linear += stride) {
        // Convert linear index to multi-dimensional indices
        int n = linear / (dim1 * dim2 * dim3);
        int rem = linear % (dim1 * dim2 * dim3);
        int c = rem / (dim2 * dim3);
        rem %= dim2 * dim3;
        int h = rem / dim3;
        int w = rem % dim3;
        int in_offset = n * in_stride0 + c * in_stride1 + h * in_stride2 + w * in_stride3;
        int out_offset = n * out_stride0 + c * out_stride1 + h * out_stride2 + w * out_stride3;
        float val = input[in_offset];
        float sig = 1.0f / (1.0f + expf(-val));
        output[out_offset] = sig;
    }
}

// InstanceNorm2d with affine=False, track_running_stats=False.
// One thread per (n,c) computes mean and variance over spatial dims (H,W),
// then normalizes all spatial elements.
__global__ void custom_instance_norm2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3,
    float eps)
{
    int nc = blockIdx.x * blockDim.x + threadIdx.x; // index over N*C
    if (nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;

    // Compute base offset for this (n,c)
    int base_in_offset = n * in_stride0 + c * in_stride1;
    int base_out_offset = n * out_stride0 + c * out_stride1;

    // Pass 1: compute mean
    float mean = 0.0f;
    int spatial_size = H * W;
    for (int hw = 0; hw < spatial_size; ++hw) {
        int h = hw / W;
        int w = hw % W;
        int in_offset = base_in_offset + h * in_stride2 + w * in_stride3;
        mean += input[in_offset];
    }
    mean /= (float)spatial_size;

    // Pass 2: compute variance (biased)
    float var = 0.0f;
    for (int hw = 0; hw < spatial_size; ++hw) {
        int h = hw / W;
        int w = hw % W;
        int in_offset = base_in_offset + h * in_stride2 + w * in_stride3;
        float diff = input[in_offset] - mean;
        var += diff * diff;
    }
    var /= (float)spatial_size;

    // Normalize and write output
    float inv_std = rsqrtf(var + eps);
    for (int hw = 0; hw < spatial_size; ++hw) {
        int h = hw / W;
        int w = hw % W;
        int in_offset = base_in_offset + h * in_stride2 + w * in_stride3;
        int out_offset = base_out_offset + h * out_stride2 + w * out_stride3;
        float normalized = (input[in_offset] - mean) * inv_std;
        output[out_offset] = normalized;
    }
}

// Launcher for custom_sigmoid
void custom_sigmoid_launcher(
    float* output,
    const float* input,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = min((total_elements + threads - 1) / threads, 65535);
    custom_sigmoid_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements,
        dim0, dim1, dim2, dim3,
        in_stride0, in_stride1, in_stride2, in_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3);
}

// Launcher for custom_instance_norm2d
void custom_instance_norm2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3,
    float eps,
    cudaStream_t stream)
{
    int total_instances = N * C;
    int threads = 256;
    int blocks = (total_instances + threads - 1) / threads;
    custom_instance_norm2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W,
        in_stride0, in_stride1, in_stride2, in_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        eps);
}
