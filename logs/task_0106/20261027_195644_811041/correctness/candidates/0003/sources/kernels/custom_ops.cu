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

// Transposed convolution kernel for 1D, with bias and output_padding (zero padding at end)
// Input: x (batch, in_channels, in_length)
// Weight: w (in_channels, out_channels, kernel_size)
// Bias: bias (out_channels)
// Output: y (batch, out_channels, out_length)
// out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding
__global__ void conv_transpose1d_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ bias,
                                        float* __restrict__ y,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int in_length,
                                        int out_length,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_padding) {
    // Each thread computes one output element: (b, oc, ol)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_outputs = (int64_t)batch * out_channels * out_length;
    int64_t stride_grid = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_outputs; i += stride_grid) {
        int b = i / (out_channels * out_length);
        int rem = i % (out_channels * out_length);
        int oc = rem / out_length;
        int ol = rem % out_length;
        float acc = 0.0f;
        // Iterate over input channels and kernel positions that contribute to this output position
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                // Input position in_length coordinate: (ol + padding - k) / stride must be integer and within bounds
                int numerator = ol + padding - k;
                if (numerator % stride != 0) continue;
                int il = numerator / stride;
                if (il >= 0 && il < in_length) {
                    // Access input x[b, ic, il]
                    float x_val = x[((int64_t)b * in_channels + ic) * in_length + il];
                    // Access weight w[ic, oc, k]
                    float w_val = w[((int64_t)ic * out_channels + oc) * kernel_size + k];
                    acc += x_val * w_val;
                }
            }
        }
        acc += bias[oc];
        y[i] = acc;
    }
}

// Launchers
void hardswish_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    hardswish_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

void softplus_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    softplus_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

void conv_transpose1d_forward_launcher(const float* x,
                                       const float* w,
                                       const float* bias,
                                       float* y,
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
    const int block_size = 256;
    int64_t total_outputs = (int64_t)batch * out_channels * out_length;
    int64_t grid_size = (total_outputs + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    conv_transpose1d_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        x, w, bias, y, batch, in_channels, out_channels, in_length, out_length,
        kernel_size, stride, padding, output_padding);
}
