#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Non-convolution CUDA kernels and launchers for model_new.py
// Conv/ConvTranspose is implemented via cuDNN in C++ binding layer
// All launchers use raw pointers and cudaStream_t

// -----------------------------------------------------------------------------
// add_bias_op: out[b,c,l] = in[b,c,l] + bias[c]
// bias shape [C,1] broadcast along L dimension
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                const float* __restrict__ bias,
                                int batch_size, int channels, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * length;
    if (idx >= total) return;

    int l = idx % length;
    int c = (idx / length) % channels;
    output[idx] = input[idx] + bias[c];
}

extern "C" void add_bias_forward_launcher(float* output,
                                          const float* input,
                                          const float* bias,
                                          int batch_size, int channels, int length,
                                          cudaStream_t stream) {
    int total = batch_size * channels * length;
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(output, input, bias,
                                                          batch_size, channels, length);
}

// -----------------------------------------------------------------------------
// softmax_channel_op: softmax over channel dimension dim=1 for each (b,l)
// input shape [B,C,L], output same shape
// Use online softmax (single pass) with max tracking and sum
// or two-pass approach. We use two-pass for simplicity and accuracy.
// -----------------------------------------------------------------------------
__global__ void softmax_channel_kernel(float* __restrict__ output,
                                       const float* __restrict__ input,
                                       int batch_size, int channels, int length) {
    int b = blockIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || l >= length) return;

    // Each thread handles one (b,l) position
    int base = b * channels * length + l;
    // Find max over channels for numerical stability
    float max_val = -1e30f;
    for (int c = 0; c < channels; ++c) {
        float v = input[base + c * length];
        if (v > max_val) max_val = v;
    }
    // Compute exp sum
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        sum += __expf(input[base + c * length] - max_val);
    }
    // Normalize and write output
    for (int c = 0; c < channels; ++c) {
        float val = __expf(input[base + c * length] - max_val) / sum;
        output[base + c * length] = val;
    }
}

extern "C" void softmax_channel_forward_launcher(float* output,
                                                 const float* input,
                                                 int batch_size, int channels, int length,
                                                 cudaStream_t stream) {
    if (batch_size <= 0 || channels <= 0 || length <= 0) return;
    // Grid dimensions: x for length, y for batch
    const int block_size_x = 256;
    int grid_x = (length + block_size_x - 1) / block_size_x;
    int grid_y = batch_size;
    dim3 block(block_size_x);
    dim3 grid(grid_x, grid_y);
    softmax_channel_kernel<<<grid, block, 0, stream>>>(output, input,
                                                       batch_size, channels, length);
}

// -----------------------------------------------------------------------------
// scale_op: out = input * scalar
// -----------------------------------------------------------------------------
__global__ void scale_kernel(float* __restrict__ output,
                             const float* __restrict__ input,
                             float scalar, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx] * scalar;
    }
}

extern "C" void scale_forward_launcher(float* output,
                                      const float* input,
                                      float scalar,
                                      int total,
                                      cudaStream_t stream) {
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    scale_kernel<<<grid_size, block_size, 0, stream>>>(output, input, scalar, total);
}
