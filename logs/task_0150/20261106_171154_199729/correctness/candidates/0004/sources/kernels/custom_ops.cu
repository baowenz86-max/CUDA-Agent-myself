// custom_ops.cu
// CUDA kernels for non-convolution ops only.
// Conv/ConvTranspose are implemented with cuDNN in C++ binding.

#include <cuda_runtime.h>

// Elementwise add with broadcast: output = input + bias
// input shape: [batch, channels, length]
// bias shape: [channels, 1]  (contiguous)
// output has same layout as input, so we assume output has same strides as input (passed in).
// We handle arbitrary strides by using 3D indexing.
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,   // stride for batch dimension (in elements)
    long long in_stride_channel, // stride for channel dimension
    long long in_stride_length,  // stride for length dimension
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * length) return;

    int l = idx % length;
    int tmp = idx / length;
    int c = tmp % channels;
    int b = tmp / channels;

    long long in_offset = b * in_stride_batch + c * in_stride_channel + l * in_stride_length;
    long long out_offset = b * out_stride_batch + c * out_stride_channel + l * out_stride_length;

    output[out_offset] = input[in_offset] + bias[c];
}

// Launcher for add_bias
extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length,
    cudaStream_t stream
) {
    int total = batch * channels * length;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(
        input, bias, output, batch, channels, length,
        in_stride_batch, in_stride_channel, in_stride_length,
        out_stride_batch, out_stride_channel, out_stride_length
    );
}
