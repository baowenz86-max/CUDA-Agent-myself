#include <cuda_runtime.h>

// Elementwise broadcast multiply: output[b,c,l] = x[b,c,l] * bias[c]
__global__ void custom_mul_kernel(const float* __restrict__ x,
                                  const float* __restrict__ bias,
                                  float* __restrict__ output,
                                  int64_t batch_size,
                                  int64_t out_channels,
                                  int64_t length_out) {
    int64_t total = batch_size * out_channels * length_out;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t c = (idx / length_out) % out_channels;
        output[idx] = x[idx] * bias[c];
    }
}

extern "C" void custom_mul_launcher(const float* x,
                                    const float* bias,
                                    float* output,
                                    int64_t batch_size,
                                    int64_t out_channels,
                                    int64_t length_out,
                                    cudaStream_t stream) {
    int64_t total = batch_size * out_channels * length_out;
    const int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    custom_mul_kernel<<<blocks, threads, 0, stream>>>(x, bias, output,
                                                      batch_size, out_channels, length_out);
}

// Transposed convolution kernel: output[b, oc, l_out] = sum_{ic, k} input[b, ic, l_out*stride + k - padding] * weight[ic, oc, k] + bias[oc]
// where valid indices satisfy 0 <= l_out*stride + k - padding < length_in.
// output_padding adds extra zeros at the end; they are not computed here (output initialized to zero by caller)
__global__ void custom_conv_transpose1d_kernel(const float* __restrict__ input,
                                               const float* __restrict__ weight,
                                               const float* __restrict__ bias,
                                               float* __restrict__ output,
                                               int batch_size,
                                               int in_channels,
                                               int out_channels,
                                               int length_in,
                                               int length_out,
                                               int kernel_size,
                                               int stride,
                                               int padding) {
    int64_t total = static_cast<int64_t>(batch_size) * out_channels * length_out;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int b = idx / (out_channels * length_out);
        int oc = (idx / length_out) % out_channels;
        int l = idx % length_out;
        float sum = bias[oc];
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_l = l * stride + k - padding;
                if (input_l >= 0 && input_l < length_in) {
                    sum += input[((b * in_channels + ic) * length_in + input_l)] *
                           weight[((ic * out_channels + oc) * kernel_size + k)];
                }
            }
        }
        output[idx] = sum;
    }
}

// Launcher for conv_transpose1d
// Output tensor must be pre-zeroed if output_padding > 0, because the kernel only fills the first
// (length_in - 1) * stride - 2 * padding + kernel_size positions; the extra output_padding positions
// remain untouched.
extern "C" void custom_conv_transpose1d_launcher(const float* input,
                                                 const float* weight,
                                                 const float* bias,
                                                 float* output,
                                                 int batch_size,
                                                 int in_channels,
                                                 int out_channels,
                                                 int length_in,
                                                 int length_out,
                                                 int kernel_size,
                                                 int stride,
                                                 int padding,
                                                 cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(batch_size) * out_channels * length_out;
    const int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    custom_conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, length_in, length_out, kernel_size, stride, padding);
}
