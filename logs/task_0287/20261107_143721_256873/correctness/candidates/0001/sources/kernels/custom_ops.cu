#include <cuda_runtime.h>
#include <cmath>

// BatchNorm2d eval mode: y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Input is NCHW layout, arbitrary strides. weight and bias may be null (treated as 1 and 0 respectively)
__global__ void batch_norm_eval_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int spatial_size,
    long long input_stride_n, long long input_stride_c,
    long long output_stride_n, long long output_stride_c)
{
    long long total = (long long)N * C * spatial_size;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int n = idx / (C * spatial_size);
        int rem = idx % (C * spatial_size);
        int c = rem / spatial_size;
        int s = rem % spatial_size;

        long long in_offset = (long long)n * input_stride_n + (long long)c * input_stride_c + s;
        long long out_offset = (long long)n * output_stride_n + (long long)c * output_stride_c + s;

        float val = input[in_offset];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        val = (val - mean) * inv_std;

        if (weight != nullptr) {
            val *= weight[c];
        }
        if (bias != nullptr) {
            val += bias[c];
        }
        output[out_offset] = val;
    }
}

// tanh forward
template <typename T>
__global__ void tanh_forward_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    long long total)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        output[idx] = tanhf((float)input[idx]);
    }
}

// Launcher for batch_norm_eval_forward
// Inputs: output, input, weight (nullable), bias (nullable), running_mean, running_var, eps,
// dims: N, C, spatial_size, input strides, output strides, stream
extern "C" void batch_norm_eval_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int spatial_size,
    long long input_stride_n, long long input_stride_c,
    long long output_stride_n, long long output_stride_c,
    cudaStream_t stream)
{
    const int block_size = 256;
    long long total = (long long)N * C * spatial_size;
    // Cap grid to avoid excessive blocks; use grid-stride loop inside kernel
    int grid_size = (int)std::min((long long)((total + block_size - 1) / block_size), (long long)65535 * 2048);
    if (grid_size <= 0) return;

    batch_norm_eval_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, eps,
        N, C, spatial_size,
        input_stride_n, input_stride_c,
        output_stride_n, output_stride_c);
}

// Launcher for tanh_forward
extern "C" void tanh_forward_launcher(
    float* output,
    const float* input,
    long long total,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (int)std::min((long long)((total + block_size - 1) / block_size), (long long)65535 * 2048);
    if (grid_size <= 0) return;

    tanh_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(output, input, total);
}
