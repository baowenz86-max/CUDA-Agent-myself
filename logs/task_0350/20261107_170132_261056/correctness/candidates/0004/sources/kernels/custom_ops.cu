#include <cuda_runtime.h>

// ReLU forward: y = max(x, 0) elementwise
extern "C" __global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

// MaxPool1d forward: kernel_size=2, stride=2, padding=0, dilation=1
// input shape: [batch, channels, length_in]
// output shape: [batch, channels, length_out]
extern "C" __global__ void max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * channels * length_out) return;

    int out_w = out_idx % length_out;
    int tmp = out_idx / length_out;
    int c = tmp % channels;
    int b = tmp / channels;

    int in_start = out_w * stride - padding;
    int in_end = in_start + kernel_size;
    if (in_start < 0) in_start = 0;
    if (in_end > length_in) in_end = length_in;

    float max_val = -3.402823466e+38f; // -FLT_MAX
    for (int in_w = in_start; in_w < in_end; ++in_w) {
        float val = input[((long long)b * channels + c) * length_in + in_w];
        if (val > max_val) max_val = val;
    }
    output[out_idx] = max_val;
}

// Conv1d forward kernel
// input: [batch, in_channels, in_length]
// weight: [out_channels, in_channels, kernel_size]
// output: [batch, out_channels, out_length]
// stride, padding
extern "C" __global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_length) return;

    int out_w = out_idx % out_length;
    int tmp = out_idx / out_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* input_ptr = input + ((long long)b * in_channels + ic) * in_length;
        const float* weight_ptr = weight + ((long long)oc * in_channels + ic) * kernel_size;
        for (int k = 0; k < kernel_size; ++k) {
            int in_w = out_w * stride - padding + k;
            if (in_w >= 0 && in_w < in_length) {
                sum += input_ptr[in_w] * weight_ptr[k];
            }
        }
    }
    output[out_idx] = sum;
}

// ConvTranspose1d forward kernel
// Equivalent to backward data of Conv1d
// input: [batch, in_channels, in_length]  (dy)
// weight: [in_channels, out_channels, kernel_size]  (filter of forward conv, but transposed conv weight shape)
//   In PyTorch ConvTranspose1d weight shape: [in_channels, out_channels/groups, kernel_size]
//   The corresponding forward conv has weight [out_channels, in_channels, kernel_size]
//   Here we use weight as given: [in_channels, out_channels, kernel_size]
// output: [batch, out_channels, out_length]  (dx)
// out_length = (in_length - 1)*stride - 2*padding + kernel_size + output_padding
extern "C" __global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_length) return;

    int out_w = out_idx % out_length;
    int tmp = out_idx / out_length;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = (bias != nullptr) ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        const float* input_ptr = input + ((long long)b * in_channels + ic) * in_length;
        const float* weight_ptr = weight + ((long long)ic * out_channels + oc) * kernel_size;
        for (int k = 0; k < kernel_size; ++k) {
            // Transposed conv relation: out_w = in_w * stride - padding + k + output_padding (if applicable)
            // Solve for in_w such that out_w + padding - k - output_padding is divisible by stride
            int numerator = out_w + padding - k;  // output_padding handled below
            if (output_padding > 0 && (out_w >= out_length - output_padding)) {
                // output_padding adds extra zeros at the end, so those positions only get bias
                // Actually output_padding just increases output size; no additional contribution from input
                // In the standard formula, output_padding is added to out_length but does not affect mapping.
                // We can ignore output_padding in mapping and simply leave those positions with bias only.
                // So skip contribution if out_w is in the output_padding region?
                // But output_padding affects the size, not the computation. We should still compute for all out_w
                // However, for out_w >= out_length - output_padding, there is no corresponding input contribution because
                // those positions are not covered by any valid in_w.
                // Actually output_padding adds extra zeros to the output, so they only get bias.
                // We'll handle by checking if the computed in_w is valid.
            }
            if (numerator < 0) continue;
            if (numerator % stride != 0) continue;
            int in_w = numerator / stride;
            if (in_w >= 0 && in_w < in_length) {
                sum += input_ptr[in_w] * weight_ptr[k];
            }
        }
    }
    output[out_idx] = sum;
}

// Launcher for ReLU
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for MaxPool1d
extern "C" void max_pool1d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream) {
    int total_out = batch_size * channels * length_out;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    max_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, length_in, length_out,
        kernel_size, stride, padding);
}

// Launcher for Conv1d
extern "C" void conv1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream) {
    int total_out = batch_size * out_channels * out_length;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    conv1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, in_length, out_length,
        kernel_size, stride, padding);
}

// Launcher for ConvTranspose1d
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream) {
    int total_out = batch_size * out_channels * out_length;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, in_length, out_length,
        kernel_size, stride, padding, output_padding);
}
