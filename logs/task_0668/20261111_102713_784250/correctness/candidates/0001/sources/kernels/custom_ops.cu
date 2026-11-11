#include <cuda_runtime.h>
#include <cstdint>

// Kernel: compute log_softmax along dimension 1 for a 4D tensor
// input shape: [batch, channels, height, width]
// Each thread processes one (batch, spatial_index).
template <int BLOCK_SIZE>
__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size)
{
    int64_t total_positions = batch * spatial_size;
    int64_t stride = (int64_t)gridDim.x * BLOCK_SIZE;
    for (int64_t pos = blockIdx.x * BLOCK_SIZE + threadIdx.x; pos < total_positions; pos += stride) {
        int64_t b = pos / spatial_size;
        int64_t s = pos - b * spatial_size;

        const float* in_base = input + b * channels * spatial_size + s;
        float* out_base = output + b * channels * spatial_size + s;

        // compute max over channels
        float max_val = in_base[0];
        #pragma unroll 4
        for (int64_t c = 1; c < channels; ++c) {
            float val = in_base[c * spatial_size];
            if (val > max_val) max_val = val;
        }

        // compute sum of exp(x - max)
        float sum_exp = 0.0f;
        #pragma unroll 4
        for (int64_t c = 0; c < channels; ++c) {
            float val = in_base[c * spatial_size];
            sum_exp += __expf(val - max_val);
        }

        float log_sum = logf(sum_exp);

        // write results
        #pragma unroll 4
        for (int64_t c = 0; c < channels; ++c) {
            float val = in_base[c * spatial_size];
            out_base[c * spatial_size] = val - max_val - log_sum;
        }
    }
}

// Launcher for log_softmax_dim1
// Expects input to be contiguous; binding layer should ensure this.
// Buffer pointers are raw device pointers.
extern "C" void log_softmax_dim1_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int64_t total_positions = batch * spatial_size;
    int blocks = (int)((total_positions + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // Limit grid size; kernel uses grid-stride loop, so any grid size works.
    if (blocks > 65535) blocks = 65535;
    log_softmax_dim1_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, batch, channels, spatial_size);
}

// Kernel: element-wise scalar division: output[i] = input[i] / divisor
template <int BLOCK_SIZE>
__global__ void scalar_div_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel,
    float divisor)
{
    int64_t stride = (int64_t)gridDim.x * BLOCK_SIZE;
    for (int64_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < numel; i += stride) {
        output[i] = input[i] / divisor;
    }
}

// Launcher for scalar_div
extern "C" void scalar_div_launcher(
    const float* input,
    float* output,
    int64_t numel,
    double divisor,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int blocks = (int)((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (blocks > 65535) blocks = 65535;
    float divisor_f = (float)divisor;
    scalar_div_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, numel, divisor_f);
}
