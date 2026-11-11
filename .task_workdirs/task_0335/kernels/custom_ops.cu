#include <cuda_runtime.h>
#include <cstdint>

// Linear forward kernel: output = input @ weight^T + bias
__global__ void linear_forward_kernel(float* __restrict__ output,
                                      const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      int batch,
                                      int in_features,
                                      int out_features) {
    int total = batch * out_features;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int b = idx / out_features;
        int o = idx % out_features;
        float sum = bias != nullptr ? bias[o] : 0.0f;
        const float* input_row = input + b * in_features;
        const float* weight_row = weight + o * in_features;
        for (int i = 0; i < in_features; ++i) {
            sum += input_row[i] * weight_row[i];
        }
        output[idx] = sum;
    }
}

// Nearest-neighbor interpolate 1D
__global__ void interpolate_nearest_1d_kernel(float* __restrict__ output,
                                               const float* __restrict__ input,
                                               int batch,
                                               int channels,
                                               int length_in,
                                               int length_out,
                                               float scale_factor) {
    int total = batch * channels * length_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int out_idx = idx % length_out;
        int c = (idx / length_out) % channels;
        int b = idx / (channels * length_out);
        int in_idx = (int)(out_idx / scale_factor);
        if (in_idx >= length_in) in_idx = length_in - 1;
        output[idx] = input[(b * channels + c) * length_in + in_idx];
    }
}

// ConvTranspose1d kernel
__global__ void conv_transpose1d_kernel(float* __restrict__ output,
                                        const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int length_in,
                                        int length_out,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_padding) {
    int total = batch * out_channels * length_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int out_pos = idx % length_out;
        int oc = (idx / length_out) % out_channels;
        int b = idx / (out_channels * length_out);

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                // Transposed convolution formula: output[b, oc, out_pos] += input[b, ic, in_pos] * weight[ic, oc, k]
                // where out_pos = in_pos * stride - padding + k, so in_pos = (out_pos + padding - k) / stride
                int in_pos_times_stride = out_pos + padding - k;
                if (in_pos_times_stride < 0 || in_pos_times_stride % stride != 0) continue;
                int in_pos = in_pos_times_stride / stride;
                if (in_pos >= 0 && in_pos < length_in) {
                    float input_val = input[(b * in_channels + ic) * length_in + in_pos];
                    float weight_val = weight[(ic * out_channels + oc) * kernel_size + k];
                    sum += input_val * weight_val;
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[idx] = sum;
    }
}

extern "C" void linear_forward_launcher(float* output,
                                        const float* input,
                                        const float* weight,
                                        const float* bias,
                                        int batch,
                                        int in_features,
                                        int out_features,
                                        cudaStream_t stream) {
    int total = batch * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    linear_forward_kernel<<<blocks, threads, 0, stream>>>(output, input, weight, bias, batch, in_features, out_features);
}

extern "C" void interpolate_nearest_1d_launcher(float* output,
                                                 const float* input,
                                                 int batch,
                                                 int channels,
                                                 int length_in,
                                                 int length_out,
                                                 float scale_factor,
                                                 cudaStream_t stream) {
    int total = batch * channels * length_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    interpolate_nearest_1d_kernel<<<blocks, threads, 0, stream>>>(output, input, batch, channels, length_in, length_out, scale_factor);
}

extern "C" void conv_transpose1d_launcher(float* output,
                                          const float* input,
                                          const float* weight,
                                          const float* bias,
                                          int batch,
                                          int in_channels,
                                          int out_channels,
                                          int length_in,
                                          int length_out,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          int output_padding,
                                          cudaStream_t stream) {
    int total = batch * out_channels * length_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(output, input, weight, bias, batch, in_channels, out_channels, length_in, length_out, kernel_size, stride, padding, output_padding);
}
