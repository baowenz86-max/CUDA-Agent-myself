#include <cuda_runtime.h>
#include <math.h>

// Element-wise sigmoid: y = 1 / (1 + exp(-x))
// Supports arbitrary strides (non-contiguous input).
__global__ void custom_sigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements,
    int ndims,
    int in_strides0, int in_strides1, int in_strides2, int in_strides3,
    int out_strides0, int out_strides1, int out_strides2, int out_strides3,
    int shape0, int shape1, int shape2, int shape3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int linear = idx; linear < total_elements; linear += stride) {
        // Convert linear index to multi-dimensional indices
        int remaining = linear;
        int in_offset = 0;
        int out_offset = 0;
        // ndims is fixed at 4 for this task, but we keep the loop for generality if ever needed
        for (int d = 0; d < ndims; ++d) {
            int dim_size;
            int in_stride;
            int out_stride;
            if (d == 0) { dim_size = shape0; in_stride = in_strides0; out_stride = out_strides0; }
            else if (d == 1) { dim_size = shape1; in_stride = in_strides1; out_stride = out_strides1; }
            else if (d == 2) { dim_size = shape2; in_stride = in_strides2; out_stride = out_strides2; }
            else { dim_size = shape3; in_stride = in_strides3; out_stride = out_strides3; }
            int coord = remaining % dim_size;
            remaining /= dim_size;
            in_offset += coord * in_stride;
            out_offset += coord * out_stride;
        }
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
    int in_strides0, int in_strides1, int in_strides2, int in_strides3,
    int out_strides0, int out_strides1, int out_strides2, int out_strides3,
    float eps)
{
    int nc = blockIdx.x * blockDim.x + threadIdx.x; // index over N*C
    if (nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;

    // Compute base offset for this (n,c)
    int base_in_offset = n * in_strides0 + c * in_strides1;
    int base_out_offset = n * out_strides0 + c * out_strides1;

    // Pass 1: compute mean
    float mean = 0.0f;
    int spatial_size = H * W;
    for (int hw = 0; hw < spatial_size; ++hw) {
        int h = hw / W;
        int w = hw % W;
        int in_offset = base_in_offset + h * in_strides2 + w * in_strides3;
        mean += input[in_offset];
    }
    mean /= (float)spatial_size;

    // Pass 2: compute variance (biased, as PyTorch InstanceNorm uses biased variance in training? Actually for evaluation with track_running_stats=False, it uses biased variance)
    float var = 0.0f;
    for (int hw = 0; hw < spatial_size; ++hw) {
        int h = hw / W;
        int w = hw % W;
        int in_offset = base_in_offset + h * in_strides2 + w * in_strides3;
        float diff = input[in_offset] - mean;
        var += diff * diff;
    }
    var /= (float)spatial_size;

    // Normalize and write output
    float inv_std = rsqrtf(var + eps);
    for (int hw = 0; hw < spatial_size; ++hw) {
        int h = hw / W;
        int w = hw % W;
        int in_offset = base_in_offset + h * in_strides2 + w * in_strides3;
        int out_offset = base_out_offset + h * out_strides2 + w * out_strides3;
        float normalized = (input[in_offset] - mean) * inv_std;
        output[out_offset] = normalized;
    }
}

// Launcher for custom_sigmoid
// Pass strides and shape as individual scalar arguments to avoid device pointer issues.
extern "C" void custom_sigmoid_launcher(
    float* output,
    const float* input,
    int total_elements,
    int ndims,
    int in_strides0, int in_strides1, int in_strides2, int in_strides3,
    int out_strides0, int out_strides1, int out_strides2, int out_strides3,
    int shape0, int shape1, int shape2, int shape3,
    cudaStream_t stream)
{
    // Use 256 threads per block, grid sized to cover elements but capped for grid-stride
    int threads = 256;
    int blocks = min((total_elements + threads - 1) / threads, 65535);
    custom_sigmoid_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements, ndims,
        in_strides0, in_strides1, in_strides2, in_strides3,
        out_strides0, out_strides1, out_strides2, out_strides3,
        shape0, shape1, shape2, shape3);
}

// Launcher for custom_instance_norm2d
extern "C" void custom_instance_norm2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int in_strides0, int in_strides1, int in_strides2, int in_strides3,
    int out_strides0, int out_strides1, int out_strides2, int out_strides3,
    float eps,
    cudaStream_t stream)
{
    int total_instances = N * C;
    int threads = 256;
    int blocks = (total_instances + threads - 1) / threads;
    custom_instance_norm2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W,
        in_strides0, in_strides1, in_strides2, in_strides3,
        out_strides0, out_strides1, out_strides2, out_strides3,
        eps);
}
