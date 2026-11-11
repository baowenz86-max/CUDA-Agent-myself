#include <cuda_runtime.h>
#include <math.h>

// Element-wise sigmoid: y = 1 / (1 + exp(-x))
// Supports arbitrary strides (non-contiguous input).
__global__ void custom_sigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements,
    int ndims,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3,
    int out_stride0, int out_stride1, int out_stride2, int out_stride3,
    int shape0, int shape1, int shape2, int shape3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int linear = idx; linear < total_elements; linear += stride) {
        int remaining = linear;
        int coord0 = remaining % shape0;
        remaining /= shape0;
        int coord1 = remaining % shape1;
        remaining /= shape1;
        int coord2 = remaining % shape2;
        remaining /= shape2;
        int coord3 = remaining % shape3;
        int in_offset = coord0 * in_stride0 + coord1 * in_stride1 + coord2 * in_stride2 + coord3 * in_stride3;
        int out_offset = coord0 * out_stride0 + coord1 * out_stride1 + coord2 * out_stride2 + coord3 * out_stride3;
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

    // Pass 2: compute variance (biased, as PyTorch InstanceNorm uses biased variance)
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
    int ndims,
    const int* in_strides,
    const int* out_strides,
    const int* shape,
    cudaStream_t stream)
{
    // Extract scalar values from host pointers (now they are host arrays, not device)
    int in_s0 = in_strides[0], in_s1 = in_strides[1], in_s2 = in_strides[2], in_s3 = in_strides[3];
    int out_s0 = out_strides[0], out_s1 = out_strides[1], out_s2 = out_strides[2], out_s3 = out_strides[3];
    int sh0 = shape[0], sh1 = shape[1], sh2 = shape[2], sh3 = shape[3];

    int threads = 256;
    int blocks = min((total_elements + threads - 1) / threads, 65535);
    custom_sigmoid_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements, ndims,
        in_s0, in_s1, in_s2, in_s3,
        out_s0, out_s1, out_s2, out_s3,
        sh0, sh1, sh2, sh3);
}

// Launcher for custom_instance_norm2d
void custom_instance_norm2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    const int* in_strides,
    const int* out_strides,
    float eps,
    cudaStream_t stream)
{
    int in_s0 = in_strides[0], in_s1 = in_strides[1], in_s2 = in_strides[2], in_s3 = in_strides[3];
    int out_s0 = out_strides[0], out_s1 = out_strides[1], out_s2 = out_strides[2], out_s3 = out_strides[3];

    int total_instances = N * C;
    int threads = 256;
    int blocks = (total_instances + threads - 1) / threads;
    custom_instance_norm2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W,
        in_s0, in_s1, in_s2, in_s3,
        out_s0, out_s1, out_s2, out_s3,
        eps);
}
