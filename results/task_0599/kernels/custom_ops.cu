// custom_ops.cu
// Contiguous float32 element-wise ops: abs and broadcast add.
// No device-side stride arrays; shapes are fixed by contract.

#include <cuda_runtime.h>
#include <cstdint>

__global__ void abs_kernel_contig(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

__global__ void add_broadcast_kernel_contig(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ param,
    int64_t numel,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t param_size = channels * height * width;
    int64_t plane_size = height * width;

    for (int64_t linear = idx; linear < numel; linear += stride) {
        // x layout: [batch, channels, height, width] contiguous
        int64_t batch_idx = linear / (channels * plane_size);
        int64_t rem = linear - batch_idx * (channels * plane_size);
        int64_t c = rem / plane_size;
        int64_t hw = rem - c * plane_size;
        // param layout: [channels, height, width] contiguous
        int64_t param_idx = c * plane_size + hw;
        output[linear] = x[linear] + param[param_idx];
    }
}

static inline int get_num_blocks(int64_t numel, int block_size) {
    int64_t blocks = (numel + block_size - 1) / block_size;
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    return (int)blocks;
}

extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {
    if (numel == 0) return;
    int block_size = 256;
    int grid_size = get_num_blocks(numel, block_size);
    abs_kernel_contig<<<grid_size, block_size, 0, stream>>>(output, input, numel);
}

extern "C" void add_scalar_tensor_forward_launcher(
    float* output,
    const float* x,
    const float* param,
    int64_t numel,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    cudaStream_t stream) {
    if (numel == 0) return;
    int block_size = 256;
    int grid_size = get_num_blocks(numel, block_size);
    add_broadcast_kernel_contig<<<grid_size, block_size, 0, stream>>>(
        output, x, param, numel, batch, channels, height, width);
}
