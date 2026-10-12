#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Helper: next power of two for grid size (not used, just for reference)
// -----------------------------------------------------------------------------
static inline int next_pow2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// -----------------------------------------------------------------------------
// Kernel 1: log10 forward
// Computes y = log10(x) for all elements.
// -----------------------------------------------------------------------------
__global__ void log10_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = log10f(input[i]);
    }
}

extern "C" void log10_launcher(const float* input,
                               float* output,
                               int64_t numel,
                               cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    int max_blocks = 1048576; // arbitrary large number, but grid.y/z not used
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(max_blocks)));
    log10_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// -----------------------------------------------------------------------------
// Kernel 2: less-than scalar (torch.lt(x, scalar))
// Output is bool, but we store as uint8_t in memory (as PyTorch does).
// -----------------------------------------------------------------------------
__global__ void lt_scalar_kernel(const float* __restrict__ input,
                                 uint8_t* __restrict__ output,
                                 float threshold,
                                 int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] < threshold) ? 1 : 0;
    }
}

extern "C" void lt_scalar_launcher(const float* input,
                                   uint8_t* output,
                                   float threshold,
                                   int64_t numel,
                                   cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    int max_blocks = 1048576;
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(max_blocks)));
    lt_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, threshold, numel);
}

// -----------------------------------------------------------------------------
// Kernel 3: boolean to float conversion (x.float())
// Since ConvTranspose1d expects float, and we can fuse this into the conv kernel,
// but the custom op expects a separate operator. We'll implement a standalone
// kernel that converts uint8_t (0/1) to float (0.0f/1.0f).
// -----------------------------------------------------------------------------
__global__ void bool_to_float_kernel(const uint8_t* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] != 0) ? 1.0f : 0.0f;
    }
}

extern "C" void bool_to_float_launcher(const uint8_t* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    int max_blocks = 1048576;
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(max_blocks)));
    bool_to_float_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// -----------------------------------------------------------------------------
// Kernel 4: ConvTranspose1d forward
// Computes output[b, oc, ow] = sum_{ic, kw} input[b, ic, iw] * weight[ic, oc, kw] + conv_bias[oc]
// where iw = (ow + padding - kw) / stride, and the division is exact when
// (ow + padding - kw) is divisible by stride and 0 <= iw < input_length.
// Additionally, after the convolution, we add the extra bias tensor (shape [out_channels, 1])
// broadcast across width. The custom op conv_transpose1d_forward should produce the
// result of ConvTranspose1d WITHOUT the extra bias, because the add_bias_forward is separate.
// So this kernel only adds conv_bias (the ConvTranspose1d bias).
// -----------------------------------------------------------------------------
__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ conv_bias,
                                        float* __restrict__ output,
                                        int batch_size,
                                        int in_channels,
                                        int out_channels,
                                        int input_length,
                                        int output_length,
                                        int kernel_size,
                                        int stride,
                                        int padding) {
    int64_t total_outputs = static_cast<int64_t>(batch_size) * out_channels * output_length;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride_total = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = idx; linear < total_outputs; linear += stride_total) {
        // Decode linear index into (b, oc, ow)
        int ow = linear % output_length;
        int64_t tmp = linear / output_length;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int numerator = ow + padding - kw;
                if (numerator % stride != 0) continue;
                int iw = numerator / stride;
                if (iw < 0 || iw >= input_length) continue;
                // input index: ((b * in_channels + ic) * input_length + iw)
                // weight index: ((ic * out_channels + oc) * kernel_size + kw)
                int64_t input_idx = ((static_cast<int64_t>(b) * in_channels + ic) * input_length + iw);
                int64_t weight_idx = (static_cast<int64_t>(ic) * out_channels + oc) * kernel_size + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
        // Add ConvTranspose1d bias
        if (conv_bias != nullptr) {
            sum += conv_bias[oc];
        }
        output[linear] = sum;
    }
}

extern "C" void conv_transpose1d_launcher(const float* input,
                                          const float* weight,
                                          const float* conv_bias,
                                          float* output,
                                          int batch_size,
                                          int in_channels,
                                          int out_channels,
                                          int input_length,
                                          int output_length,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          cudaStream_t stream) {
    const int threads = 256;
    int64_t total_outputs = static_cast<int64_t>(batch_size) * out_channels * output_length;
    int64_t blocks64 = (total_outputs + threads - 1) / threads;
    int max_blocks = 1048576;
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(max_blocks)));
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, conv_bias, output,
        batch_size, in_channels, out_channels, input_length, output_length,
        kernel_size, stride, padding);
}

// -----------------------------------------------------------------------------
// Kernel 5: Add bias (shape [out_channels, 1]) broadcast across width
// Computes output[b, oc, ow] = input[b, oc, ow] + bias[oc]
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int batch_size,
                                int channels,
                                int width) {
    int64_t total_elements = static_cast<int64_t>(batch_size) * channels * width;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int ow = linear % width;
        int64_t tmp = linear / width;
        int oc = tmp % channels;
        // b = tmp / channels;
        output[linear] = input[linear] + bias[oc];
    }
}

extern "C" void add_bias_launcher(const float* input,
                                  const float* bias,
                                  float* output,
                                  int batch_size,
                                  int channels,
                                  int width,
                                  cudaStream_t stream) {
    const int threads = 256;
    int64_t total_elements = static_cast<int64_t>(batch_size) * channels * width;
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    int max_blocks = 1048576;
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(max_blocks)));
    add_bias_kernel<<<blocks, threads, 0, stream>>>(
        input, bias, output, batch_size, channels, width);
}
