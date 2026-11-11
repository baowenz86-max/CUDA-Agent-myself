// custom_ops.cu: CUDA kernels for cat_maximum_forward
// This file contains only CUDA kernels and host launchers.
// It does NOT include any PyTorch headers or Tensor wrappers.
#include <cuda_runtime.h>
#include <cstdint>

// Kernel:
// output = torch.cat((input, input), dim=1), then torch.maximum(output, output)
// which simplifies to just the concatenation.
// The kernel handles arbitrary strides and large dimensions using 64-bit indexing.
// Grid-stride loop ensures every output element is written.
__global__ void cat_maximum_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int in_channels,
    int height,
    int width,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t in_h_stride,
    int64_t in_w_stride,
    int64_t out_n_stride,
    int64_t out_c_stride,
    int64_t out_h_stride,
    int64_t out_w_stride)
{
    int64_t out_total = (int64_t)batch * (2 * in_channels) * height * width;
    int64_t total_threads = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < out_total;
         idx += total_threads)
    {
        // Decompose linear index into n, c, h, w
        int64_t rem = idx;
        int64_t w = rem % width;
        rem /= width;
        int64_t h = rem % height;
        rem /= height;
        int64_t c = rem % (2 * in_channels);
        int64_t n = rem / (2 * in_channels);

        // Input channel: concatenation duplicates, so use c modulo in_channels
        int64_t in_c = c % in_channels;

        // Compute input and output linear offsets using strides
        int64_t in_offset = n * in_n_stride + in_c * in_c_stride + h * in_h_stride + w * in_w_stride;
        int64_t out_offset = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;

        output[out_offset] = input[in_offset];
    }
}

// Host launcher with C linkage for binding.cpp to call.
// Parameters are raw pointers and scalar metadata; no tensor objects.
extern "C" void cat_maximum_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int in_channels,
    int height,
    int width,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t in_h_stride,
    int64_t in_w_stride,
    int64_t out_n_stride,
    int64_t out_c_stride,
    int64_t out_h_stride,
    int64_t out_w_stride,
    cudaStream_t stream)
{
    int64_t out_total = (int64_t)batch * (2 * in_channels) * height * width;
    if (out_total <= 0) return;

    // Choose block size 256, grid limited to 65535 in y but use 1D limited to 2^31-1 blocks,
    // so use 256 threads and compute grid to cover all elements with grid-stride.
    int block_size = 256;
    int64_t max_blocks = 2147483647; // 2^31 - 1, safe for 1D grid.x
    int64_t blocks64 = (out_total + block_size - 1) / block_size;
    if (blocks64 > max_blocks) blocks64 = max_blocks;
    int grid_size = (int)blocks64;

    cat_maximum_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input,
        batch, in_channels, height, width,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride);
}
