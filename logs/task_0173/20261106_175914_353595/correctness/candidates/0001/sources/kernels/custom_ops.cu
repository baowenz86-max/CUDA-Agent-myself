#include <cuda_runtime.h>
#include <cmath>

// Kernel: einsum 'bhwc->bwh'
// output shape: [batch, width, height]
// input shape: [batch, height, width, channels]
// For each output element (b, w, h), sum_{c=0}^{C-1} input[b, h, w, c]
__global__ void einsum_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int height,
    int width,
    int channels,
    long long input_batch_stride,
    long long input_height_stride,
    long long input_width_stride,
    long long input_channel_stride)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * width * height) return;

    int b = out_idx / (width * height);
    int rem = out_idx - b * (width * height);
    int w = rem / height;
    int h = rem - w * height;

    const float* base = input + b * input_batch_stride + h * input_height_stride + w * input_width_stride;
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        sum += base[c * input_channel_stride];
    }
    output[out_idx] = sum;
}

// Kernel: elementwise absolute value
__global__ void abs_kernel(float* __restrict__ output, const float* __restrict__ input, long long num_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fabsf(input[idx]);
    }
}

// Launchers
extern "C" void einsum_bhwc_to_bwh_launcher(
    float* output,
    const float* input,
    int batch,
    int height,
    int width,
    int channels,
    long long input_batch_stride,
    long long input_height_stride,
    long long input_width_stride,
    long long input_channel_stride,
    cudaStream_t stream)
{
    long long total = (long long)batch * width * height;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    einsum_kernel<<<grid, block, 0, stream>>>(
        output, input, batch, height, width, channels,
        input_batch_stride, input_height_stride, input_width_stride, input_channel_stride);
}

extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    long long num_elements,
    cudaStream_t stream)
{
    int block = 256;
    long long grid_ll = (num_elements + block - 1) / block;
    int grid = (int)grid_ll;
    abs_kernel<<<grid, block, 0, stream>>>(output, input, num_elements);
}
