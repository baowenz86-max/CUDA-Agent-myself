#include <cuda_runtime.h>

// Element-wise absolute value kernel.
// Each thread processes one element. Grid-stride loop supports arbitrary numel.
__global__ void abs_kernel(const float* input, float* output, long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = fabsf(input[idx]);
    }
}

// Adaptive average pooling over the last dimension (width).
// Assumes input_width is divisible by output_width (window_size = input_width / output_width).
// Each thread computes one output element by averaging window_size consecutive inputs.
__global__ void adaptive_avg_pool1d_kernel(const float* input, float* output,
                                          int batch, int channels,
                                          int input_width, int output_width,
                                          int window_size) {
    int total_outputs = batch * channels * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;

    // Decode output index: [b][c][ow]
    int ow = idx % output_width;
    int tmp = idx / output_width;
    int c = tmp % channels;
    int b = tmp / channels;

    // Input start offset for this window
    int in_start = ((b * channels + c) * input_width) + ow * window_size;

    float sum = 0.0f;
    for (int i = 0; i < window_size; ++i) {
        sum += input[in_start + i];
    }
    output[idx] = sum / (float)window_size;
}

// Launcher for custom_abs_forward.
extern "C" void custom_abs_forward_launcher(const float* input, float* output,
                                             long long numel, cudaStream_t stream) {
    const int block_size = 256;
    const long long grid_size = (numel + block_size - 1) / block_size;
    // Grid dimension limited to 2^31-1, but numel here is 104857600, well below limit.
    abs_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

// Launcher for custom_adaptive_avg_pool1d_forward.
extern "C" void custom_adaptive_avg_pool1d_forward_launcher(
        const float* input, float* output,
        int batch, int channels, int input_width, int output_width,
        int window_size, cudaStream_t stream) {
    int total_outputs = batch * channels * output_width;
    const int block_size = 256;
    const int grid_size = (total_outputs + block_size - 1) / block_size;
    adaptive_avg_pool1d_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, batch, channels, input_width, output_width, window_size);
}
