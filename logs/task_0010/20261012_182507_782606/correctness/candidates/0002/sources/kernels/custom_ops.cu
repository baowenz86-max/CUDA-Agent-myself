#include <cuda_runtime.h>
#include <cstdint>

// Define macros for launch bounds
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// -----------------------------------------------------------------------------
// Convolution kernel (direct, float32)
// -----------------------------------------------------------------------------
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_h, int kernel_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    if (idx >= total_elements) return;

    // Decode output position
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);

    float sum = bias[oc];

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh + kh;  // stride 1, padding 0
            if (ih < 0 || ih >= in_height) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = ow + kw;
                if (iw < 0 || iw >= in_width) continue;
                int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[idx] = sum;
}

// -----------------------------------------------------------------------------
// Triu kernel: zero elements below main diagonal for each (batch, channel) matrix
// -----------------------------------------------------------------------------
__global__ void triu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * height * width;
    if (idx >= total) return;

    int w = idx % width;
    int h = (idx / width) % height;
    // batch and channel determine the matrix but are irrelevant for diagonal condition
    if (h <= w) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Bitwise XOR with broadcast bias (uint8)
// -----------------------------------------------------------------------------
__global__ void bitwise_xor_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ output,
    int batch_size, int channels, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * height * width;
    if (idx >= total) return;

    int c = (idx / (height * width)) % channels;
    output[idx] = input[idx] ^ bias[c];
}

// -----------------------------------------------------------------------------
// Cast uint8 to float32
// -----------------------------------------------------------------------------
__global__ void cast_float_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int total) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = static_cast<float>(input[idx]);
    }
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C" for binding)
// -----------------------------------------------------------------------------
extern "C" void conv2d_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_h, int kernel_w,
    cudaStream_t stream) {

    int total = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_h, kernel_w);
}

extern "C" void triu_forward_launcher(
    const float* input, float* output,
    int batch_size, int channels, int height, int width,
    cudaStream_t stream) {

    int total = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    triu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, height, width);
}

extern "C" void bitwise_xor_forward_launcher(
    const uint8_t* input, const uint8_t* bias, uint8_t* output,
    int batch_size, int channels, int height, int width,
    cudaStream_t stream) {

    int total = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bitwise_xor_kernel<<<blocks, threads, 0, stream>>>(
        input, bias, output, batch_size, channels, height, width);
}

extern "C" void cast_float_forward_launcher(
    const uint8_t* input, float* output,
    int total,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cast_float_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
