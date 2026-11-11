#include <cuda_runtime.h>

// Adaptive average pooling over the last dimension.
// Each thread computes one output element.
// Window indices follow PyTorch: start = floor(i*L/O), end = ceil((i+1)*L/O)
__global__ void adaptive_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long batch,
    long long channels,
    long long seq_len,
    long long output_size) {

    long long total_outputs = batch * channels * output_size;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (; idx < total_outputs; idx += stride) {
        long long out_idx = idx % output_size;
        long long ch = (idx / output_size) % channels;
        long long b = idx / (channels * output_size);

        long long start = (out_idx * seq_len) / output_size;
        long long end = ((out_idx + 1) * seq_len + output_size - 1) / output_size; // ceil
        if (end > seq_len) end = seq_len;

        float sum = 0.0f;
        const float* in_base = input + (b * channels + ch) * seq_len;
        for (long long i = start; i < end; ++i) {
            sum += in_base[i];
        }
        long long count = end - start;
        if (count > 0) {
            output[idx] = sum / static_cast<float>(count);
        } else {
            output[idx] = 0.0f;
        }
    }
}

extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    long long batch,
    long long channels,
    long long seq_len,
    long long output_size,
    cudaStream_t stream) {

    long long total_outputs = batch * channels * output_size;
    const int block_size = 256;
    long long blocks = (total_outputs + block_size - 1) / block_size;
    if (blocks < 1) blocks = 1;

    adaptive_avg_pool1d_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input, output, batch, channels, seq_len, output_size);
}
