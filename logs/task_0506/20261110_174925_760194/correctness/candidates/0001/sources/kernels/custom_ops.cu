#include <cuda_runtime.h>

// Elementwise ReLU: output = max(input, 0)
__global__ void relu_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

// MaxPool1d along last dimension with non-overlapping windows
// Input shape: [batch, channels, length], output shape: [batch, channels, out_length]
// Each output element = max over window of size pool_size
__global__ void max_pool1d_forward_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int64_t batch_size,
                                          int64_t channels,
                                          int64_t in_length,
                                          int64_t out_length,
                                          int64_t pool_size) {
    int64_t total_outputs = batch_size * channels * out_length;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t out_idx = idx; out_idx < total_outputs; out_idx += stride) {
        // Decompose output linear index into (b, c, o)
        int64_t o = out_idx % out_length;
        int64_t bc = out_idx / out_length;
        int64_t c = bc % channels;
        int64_t b = bc / channels;

        // Input start offset for this output position
        int64_t in_start = ((b * channels + c) * in_length) + o * pool_size;
        float max_val = -3.402823466e+38f; // -FLT_MAX
        // Pool window may be partially empty if pool_size does not divide in_length,
        // but contract says trailing elements are ignored, so we only consider full window.
        // However, to be safe, loop over all pool_size (window should be fully within bounds
        // when out_length = floor(in_length / pool_size) and o < out_length).
        #pragma unroll 8
        for (int64_t k = 0; k < pool_size; ++k) {
            float val = input[in_start + k];
            if (val > max_val) max_val = val;
        }
        output[out_idx] = max_val;
    }
}

// Launcher for ReLU
extern "C" void relu_forward_launcher(const float* input,
                                     float* output,
                                     int64_t total_elements,
                                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit for large tensors
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// Launcher for MaxPool1d
extern "C" void max_pool1d_forward_launcher(const float* input,
                                            float* output,
                                            int64_t batch_size,
                                            int64_t channels,
                                            int64_t in_length,
                                            int64_t out_length,
                                            int64_t pool_size,
                                            cudaStream_t stream) {
    int threads = 256;
    int64_t total_outputs = batch_size * channels * out_length;
    int blocks = (int)((total_outputs + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_pool1d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, in_length, out_length, pool_size);
}
