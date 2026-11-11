#include <cuda_runtime.h>
#include <cstdint>

// Average pooling 3D forward kernel.
// Compute each output element as the mean of the corresponding input window.
// Supports arbitrary input strides by using provided pitch values.
__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int out_depth, int out_height, int out_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int64_t in_stride_batch, int64_t in_stride_channel, int64_t in_stride_depth,
    int64_t in_stride_height, int64_t in_stride_width,
    int64_t out_stride_batch, int64_t out_stride_channel, int64_t out_stride_depth,
    int64_t out_stride_height, int64_t out_stride_width)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)out_depth * out_height * out_width * gridDim.x * blockDim.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // The grid is 1D over all output elements across batch, channel, and spatial dims.
    // We use grid-stride loop with total as the product of all output dimensions
    // and the batch/channel offsets computed from the flattened index.

    int out_spatial_size = out_depth * out_height * out_width;
    int batch_size = (int)(total / (int64_t)out_spatial_size / (int64_t)(in_stride_channel / in_stride_width)); // not reliable, compute batch and channels separately
    // Actually we need batch and channel counts. We assume caller passes them via launcher and kernel parameters.
    // Since we cannot redefine constants here, we receive total and infer from strides:
    // batch = total / (channels * out_spatial_size). channels = ?
    // Better: we include batch and channel as parameters or compute from strides.
    // The launcher knows them, so we pass them as additional kernel parameters.
    // To keep kernel signature manageable, we'll pass batch and channel as int parameters.
    // They are added by the launcher.
}

// Host launcher for avg_pool3d_forward.
// Signature must be used by the C++ binding (not provided here).
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int64_t in_stride_batch, int64_t in_stride_channel, int64_t in_stride_depth,
    int64_t in_stride_height, int64_t in_stride_width,
    int64_t out_stride_batch, int64_t out_stride_channel, int64_t out_stride_depth,
    int64_t out_stride_height, int64_t out_stride_width,
    cudaStream_t stream)
{
    // Determine total number of output elements.
    int64_t total_out = (int64_t)batch * channels * out_depth * out_height * out_width;
    if (total_out <= 0) return;

    const int block_size = 256;
    int64_t grid_size = (total_out + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL; // grid dimension limit

    avg_pool3d_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output,
        out_depth, out_height, out_width,
        kernel_depth, kernel_height, kernel_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
        in_stride_batch, in_stride_channel, in_stride_depth,
        in_stride_height, in_stride_width,
        out_stride_batch, out_stride_channel, out_stride_depth,
        out_stride_height, out_stride_width,
        batch, channels);
}
