#include <cuda_runtime.h>

// Elementwise Hardswish: y = x * min(max(x + 3, 0), 6) / 6
__global__ void hardswish_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[i] = x * relu6 * (1.0f / 6.0f);
    }
}

// Elementwise Softplus: y = log(1 + exp(x)) for x <= 20, else x
__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// Custom transposed convolution kernel for ConvTranspose1d
// Parameters:
// input: (batch, in_channels, in_length)
// weight: (in_channels, out_channels, kernel_size)
// output: (batch, out_channels, out_length)
// stride, padding, output_padding as ints
// kernel_size is weight.size(2)
__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int in_length,
                                        int out_length,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_padding) {
    int total_elements = batch * out_channels * out_length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int out_position = idx % out_length;
    int out_channel = (idx / out_length) % out_channels;
    int batch_idx = idx / (out_length * out_channels);

    float sum = 0.0f;
    // Iterate over input positions that contribute to this output
    for (int kh = 0; kh < kernel_size; ++kh) {
        // formula: out_position = in_position * stride - padding + kh
        // => in_position = (out_position + padding - kh) / stride, must be integer and within range
        int numerator = out_position + padding - kh;
        if (numerator < 0) continue;
        if (numerator % stride != 0) continue;
        int in_position = numerator / stride;
        if (in_position >= in_length) continue;
        // Accumulate over all input channels
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            float input_val = input[((batch_idx * in_channels + in_channel) * in_length) + in_position];
            float weight_val = weight[((in_channel * out_channels + out_channel) * kernel_size) + kh];
            sum += input_val * weight_val;
        }
    }
    // Add bias
    sum += bias[out_channel];
    output[idx] = sum;
}

// Launcher for hardswish elementwise op
void hardswish_forward_launcher(const float* input,
                                float* output,
                                int64_t numel,
                                cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    hardswish_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handled in binding
    }
}

// Launcher for softplus elementwise op
void softplus_forward_launcher(const float* input,
                               float* output,
                               int64_t numel,
                               cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    softplus_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handled in binding
    }
}

// Launcher for custom transposed convolution
void conv_transpose1d_launcher(const float* input,
                               const float* weight,
                               const float* bias,
                               float* output,
                               int batch,
                               int in_channels,
                               int out_channels,
                               int in_length,
                               int out_length,
                               int kernel_size,
                               int stride,
                               int padding,
                               int output_padding,
                               cudaStream_t stream) {
    int total_elements = batch * out_channels * out_length;
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    conv_transpose1d_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, in_length, out_length, kernel_size, stride, padding, output_padding);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handled in binding
    }
}
