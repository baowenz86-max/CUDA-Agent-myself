#include <cuda_runtime.h>

// CircularPad1d kernel: pads the last dimension circularly.
// Input: (batch, channels, length) float32
// Output: (batch, channels, length + 2*padding) float32
__global__ void circular_pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t length,
    int64_t padding,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t out_length = length + 2 * padding;
    int64_t plane_size = channels * out_length;
    int64_t input_plane_size = channels * length;

    for (int64_t i = idx; i < total_elements; i += stride) {
        // Decompose linear index into batch, channel, and position within padded length
        int64_t batch_idx = i / plane_size;
        int64_t rem = i - batch_idx * plane_size;
        int64_t channel_idx = rem / out_length;
        int64_t out_pos = rem - channel_idx * out_length;

        int64_t in_pos;
        if (out_pos < padding) {
            // left padding: take from end of input
            in_pos = length - padding + out_pos;
        } else if (out_pos >= padding + length) {
            // right padding: take from beginning of input
            in_pos = out_pos - (padding + length);
        } else {
            // main copy
            in_pos = out_pos - padding;
        }

        // Compute input offset
        int64_t input_idx = batch_idx * input_plane_size + channel_idx * length + in_pos;
        output[i] = input[input_idx];
    }
}

// Launcher function with C linkage, to be called from binding.cpp
extern "C" void circular_pad_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t length,
    int64_t padding,
    cudaStream_t stream)
{
    int64_t out_length = length + 2 * padding;
    int64_t total_elements = batch * channels * out_length;

    const int block_size = 256;
    // Compute grid size: enough blocks to cover total_elements with grid-stride loop
    // Use at most 2^31-1 blocks, but typical total elements may require multiple waves.
    int64_t blocks_64 = (total_elements + block_size - 1) / block_size;
    int max_blocks = 65535; // safe upper bound for most GPUs, but we can use larger.
    int blocks = static_cast<int>(blocks_64 > max_blocks ? max_blocks : blocks_64);
    if (blocks == 0) blocks = 1; // in case total_elements == 0? but we may have zero? no, batch,channels,length positive.

    circular_pad_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, batch, channels, length, padding, total_elements);
}
