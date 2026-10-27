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

// ConvTranspose1d kernel: output[b, oc, ol] = bias[oc] + sum_{ic, k} input[b, ic, il] * weight[ic, oc, k]
// where ol = il * stride - padding + k, and output_padding adds zeros at the end.
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
                                        int padding) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)batch * out_channels * out_length;
    if (idx >= total) return;

    int ol = idx % out_length;
    int oc = (idx / out_length) % out_channels;
    int b = idx / ((int64_t)out_channels * out_length);

    float sum = bias[oc];
    int il_min = max(0, (ol + padding - (kernel_size - 1) + stride - 1) / stride);
    int il_max = min(in_length - 1, (ol + padding) / stride);

    for (int il = il_min; il <= il_max; ++il) {
        int k = ol - il * stride + padding;
        if (k >= 0 && k < kernel_size) {
            for (int ic = 0; ic < in_channels; ++ic) {
                float in_val = input[((int64_t)b * in_channels + ic) * in_length + il];
                float w_val = weight[((int64_t)ic * out_channels + oc) * kernel_size + k];
                sum += in_val * w_val;
            }
        }
    }
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
}

// Launcher for conv_transpose1d op
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
                               cudaStream_t stream) {
    int64_t total = (int64_t)batch * out_channels * out_length;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    conv_transpose1d_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, weight, bias, output, batch, in_channels, out_channels, in_length, out_length, kernel_size, stride, padding);
}
