#include <cuda_runtime.h>

// ReLU forward kernel: elementwise max(input, 0)
// Supports arbitrary strides; each thread processes one element via grid-stride loop.
__global__ void relu_forward_kernel(
    float* output,
    const float* input,
    int64_t N,
    int64_t input_offset,
    int64_t output_offset)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        int64_t in_idx = input_offset + idx;
        int64_t out_idx = output_offset + idx;
        float val = input[in_idx];
        output[out_idx] = val > 0.0f ? val : 0.0f;
    }
}

// BatchNorm eval kernel for input shape [N, C, W] (or any dims with channel at dim1).
// Computes: out[n,c,w] = (in[n,c,w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Supports arbitrary strides via explicit indexing.
__global__ void batch_norm_eval_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t outer_size,   // N (batches)
    int64_t C,            // channels
    int64_t inner_size,   // W (elements per channel per batch)
    int64_t in_stride_outer,
    int64_t in_stride_channel,
    int64_t in_stride_inner,
    int64_t out_stride_outer,
    int64_t out_stride_channel,
    int64_t out_stride_inner)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outer_size * C * inner_size; idx += blockDim.x * gridDim.x) {
        // decompose idx into outer, channel, inner
        int64_t inner = idx % inner_size;
        int64_t tmp = idx / inner_size;
        int64_t channel = tmp % C;
        int64_t outer = tmp / C;

        int64_t in_idx = outer * in_stride_outer + channel * in_stride_channel + inner * in_stride_inner;
        int64_t out_idx = outer * out_stride_outer + channel * out_stride_channel + inner * out_stride_inner;

        float mean = running_mean[channel];
        float var = running_var[channel];
        float w = weight[channel];
        float b = bias[channel];
        float inv_std = rsqrtf(var + eps);

        float val = (input[in_idx] - mean) * inv_std * w + b;
        output[out_idx] = val;
    }
}

// Host launcher for ReLU.
// Accepts raw pointers, dimensions, strides, and CUDA stream.
// output_ptr must be a valid allocated buffer with same number of elements as input.
extern "C" void relu_forward_launcher(
    float* output_ptr,
    const float* input_ptr,
    int64_t numel,
    int64_t input_offset,
    int64_t output_offset,
    cudaStream_t stream)
{
    // Use 256 threads per block, enough blocks to cover the data with grid-stride loop.
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size; grid-stride loop handles remainder.
    if (blocks > 65535) blocks = 65535;

    relu_forward_kernel<<<blocks, threads, 0, stream>>>(
        output_ptr, input_ptr, numel, input_offset, output_offset);
}

// Host launcher for BatchNorm eval.
// Expects input shape [outer_size, C, inner_size] (with outer possibly batch dims,
// but here outer_size corresponds to N, C = num_features, inner_size = W).
// output_ptr must be allocated with same shape.
extern "C" void batch_norm_eval_launcher(
    float* output_ptr,
    const float* input_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    const float* running_mean_ptr,
    const float* running_var_ptr,
    float eps,
    int64_t outer_size,
    int64_t C,
    int64_t inner_size,
    int64_t in_stride_outer,
    int64_t in_stride_channel,
    int64_t in_stride_inner,
    int64_t out_stride_outer,
    int64_t out_stride_channel,
    int64_t out_stride_inner,
    cudaStream_t stream)
{
    int64_t total_elements = outer_size * C * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    batch_norm_eval_kernel<<<blocks, threads, 0, stream>>>(
        output_ptr, input_ptr, weight_ptr, bias_ptr,
        running_mean_ptr, running_var_ptr, eps,
        outer_size, C, inner_size,
        in_stride_outer, in_stride_channel, in_stride_inner,
        out_stride_outer, out_stride_channel, out_stride_inner);
}
