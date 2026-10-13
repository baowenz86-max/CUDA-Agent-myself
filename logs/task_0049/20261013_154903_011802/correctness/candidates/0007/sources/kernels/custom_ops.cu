#include <cuda_runtime.h>

// CUDA kernel for add_channel_bias_round:
// out[b,c,l] = roundf(x[b,c,l] + bias[c])
// x may have arbitrary strides, but dims: batches, channels, length.
// bias is contiguous with length channels.
__global__ void add_channel_bias_round_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    int batches,
    int channels,
    int length,
    int out_batch_stride,
    int out_channel_stride,
    int in_batch_stride,
    int in_channel_stride) {
    int total = batches * channels * length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int l = i % length;
        int tmp = i / length;
        int c = tmp % channels;
        int b = tmp / channels;
        int out_idx = b * out_batch_stride + c * out_channel_stride + l;
        int in_idx = b * in_batch_stride + c * in_channel_stride + l;
        out[out_idx] = roundf(x[in_idx] + bias[c]);
    }
}

// Launcher for add_channel_bias_round.
// Inputs:
//   out: output tensor pointer (must be preallocated, same shape as x)
//   x: input tensor pointer
//   bias: bias tensor pointer (contiguous, length = channels)
//   batches, channels, length: tensor dimensions
//   out_batch_stride, out_channel_stride: strides of output tensor (element strides)
//   in_batch_stride, in_channel_stride: strides of input tensor
//   stream: CUDA stream
void add_channel_bias_round_launcher(
    float* out,
    const float* x,
    const float* bias,
    int batches,
    int channels,
    int length,
    int out_batch_stride,
    int out_channel_stride,
    int in_batch_stride,
    int in_channel_stride,
    cudaStream_t stream) {
    int total = batches * channels * length;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Limit grid size to avoid excessive blocks
    if (grid_size > 65535) grid_size = 65535;
    add_channel_bias_round_kernel<<<grid_size, block_size, 0, stream>>>(
        out, x, bias, batches, channels, length,
        out_batch_stride, out_channel_stride,
        in_batch_stride, in_channel_stride);
}
