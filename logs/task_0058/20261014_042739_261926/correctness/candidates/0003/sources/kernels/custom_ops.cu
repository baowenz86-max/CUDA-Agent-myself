#include <cuda_runtime.h>

// Kernel: sum over dimension 1 (channels) for 3D tensor (batch, channels, length)
// Output shape: (batch, length)
// Supports arbitrary strides for input.
__global__ void sum_dim1_kernel(float* __restrict__ output, const float* __restrict__ input,
                                int batch, int channels, int length,
                                long long in_stride_batch, long long in_stride_channel, long long in_stride_length) {
    // Each thread computes one output element: (b, l)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * length;
    if (idx >= total) return;

    int b = idx / length;
    int l = idx - b * length;

    float sum = 0.0f;
    const float* base = input + b * in_stride_batch + l * in_stride_length;
    for (int c = 0; c < channels; ++c) {
        sum += base[c * in_stride_channel];
    }
    output[idx] = sum;
}

// Launcher: sum_dim_forward
extern "C" void sum_dim_launcher(float* output, const float* input,
                                 int dim, int batch, int channels, int length,
                                 long long in_stride_batch, long long in_stride_channel, long long in_stride_length,
                                 long long out_stride_batch, long long out_stride_length,
                                 cudaStream_t stream) {
    if (dim != 1) {
        return; // Only dim=1 supported
    }

    int total = batch * length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(
        output, input, batch, channels, length,
        in_stride_batch, in_stride_channel, in_stride_length);
}

// Kernel: 1D convolution (forward) with bias, supports arbitrary stride/padding.
// Input: [batch, in_channels, length]
// Weight: [out_channels, in_channels, kernel_size]
// Output: [batch, out_channels, output_length]
// We assume input and weight are contiguous, output is contiguous.
__global__ void conv1d_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                              const float* __restrict__ bias, float* __restrict__ output,
                              int batch, int in_channels, int out_channels, int length,
                              int kernel_size, int stride, int padding, int output_length) {
    // Each thread computes one output element: (b, oc, ol)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * output_length;
    if (idx >= total) return;

    int ol = idx % output_length;
    int tmp = idx / output_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        const float* input_ptr = input + ((b * in_channels + ic) * length);
        const float* weight_ptr = weight + ((oc * in_channels + ic) * kernel_size);
        for (int k = 0; k < kernel_size; ++k) {
            int input_index = ol * stride - padding + k;
            if (input_index >= 0 && input_index < length) {
                sum += input_ptr[input_index] * weight_ptr[k];
            }
        }
    }

    output[idx] = sum;
}

// Launcher for conv1d
extern "C" void conv1d_launcher(const float* input, const float* weight, const float* bias,
                                float* output, int batch, int in_channels, int out_channels, int length,
                                int kernel_size, int stride, int padding, int output_length,
                                cudaStream_t stream) {
    int total = batch * out_channels * output_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, batch, in_channels, out_channels, length,
        kernel_size, stride, padding, output_length);
}

// Kernel: 1D transposed convolution (backward data) with bias.
// This is equivalent to ConvTranspose1d forward.
// Input: [batch, in_channels, length_in]
// Weight: [in_channels, out_channels, kernel_size] (PyTorch layout)
// Output: [batch, out_channels, length_out]
// length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding
__global__ void conv_transpose1d_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                        const float* __restrict__ bias, float* __restrict__ output,
                                        int batch, int in_channels, int out_channels, int length_in,
                                        int kernel_size, int stride, int padding, int output_padding,
                                        int length_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * length_out;
    if (idx >= total) return;

    int ol = idx % length_out;
    int tmp = idx / length_out;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        const float* input_ptr = input + (b * in_channels + ic) * length_in;
        const float* weight_ptr = weight + (ic * out_channels + oc) * kernel_size;
        for (int k = 0; k < kernel_size; ++k) {
            // Transposed conv relation: output[ol] += input[il] * weight[k] where ol = il*stride + k - padding
            // So to compute output[ol], we need input index il such that il*stride + k - padding = ol
            // => il = (ol + padding - k) / stride, and we need (ol + padding - k) % stride == 0
            int numerator = ol + padding - k;
            if (numerator < 0 || numerator % stride != 0) continue;
            int il = numerator / stride;
            if (il >= 0 && il < length_in) {
                sum += input_ptr[il] * weight_ptr[k];
            }
        }
    }

    output[idx] = sum;
}

// Launcher for conv_transpose1d
extern "C" void conv_transpose1d_launcher(const float* input, const float* weight, const float* bias,
                                          float* output, int batch, int in_channels, int out_channels, int length_in,
                                          int kernel_size, int stride, int padding, int output_padding,
                                          int length_out, cudaStream_t stream) {
    int total = batch * out_channels * length_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, batch, in_channels, out_channels, length_in,
        kernel_size, stride, padding, output_padding, length_out);
}
