#include <cuda_runtime.h>

// ConvTranspose1d kernel
// Input: (batch, in_channels, length_in)
// Weight: (in_channels, out_channels, kernel_size)
// Bias: (out_channels)
// Output: (batch, out_channels, length_out)
// length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding
// All tensors assumed contiguous.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
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
    int padding,
    int output_padding)
{
    int total = batch_size * out_channels * length_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride_total) {
        int l_out = i % length_out;
        int tmp = i / length_out;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* input_ptr = input + (b * in_channels + ic) * length_in;
            const float* weight_ptr = weight + (ic * out_channels + oc) * kernel_size;
            for (int k = 0; k < kernel_size; ++k) {
                int pos = l_out + padding - k;
                if (pos >= 0 && pos % stride == 0) {
                    int l_in = pos / stride;
                    if (l_in < length_in) {
                        sum += input_ptr[l_in] * weight_ptr[k];
                    }
                }
            }
        }
        output[i] = sum + bias[oc];
    }
}

// Sign kernel: elementwise sign, for float32
extern "C" void conv_transpose1d_forward(
    const float* input,
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
    int output_padding,
    cudaStream_t stream)
{
    int total = batch_size * out_channels * length_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap, kernel uses grid-stride loop
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, length_in, length_out,
        kernel_size, stride, padding, output_padding);
}

__global__ void sign_kernel(const float* __restrict__ input, float* __restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float v = input[i];
        output[i] = (v > 0.0f) - (v < 0.0f);
    }
}

extern "C" void sign_forward(const float* input, float* output, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sign_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// Embedding kernel: gather rows from weight
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_indices,
    int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_indices; i += stride) {
        int64_t index = indices[i];
        // Assume indices are valid (< num_embeddings); binding can check but we don't have num_embeddings here.
        // We'll rely on weight rows count provided.
        // Use float4 for vectorized copy if possible
        const float* src = weight + index * embedding_dim;
        float* dst = output + i * embedding_dim;
        // copy embedding_dim floats
        for (int j = 0; j < embedding_dim; ++j) {
            dst[j] = src[j];
        }
    }
}

extern "C" void embedding_forward(
    const int64_t* indices,
    const float* weight,
    float* output,
    int num_indices,
    int embedding_dim,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_indices + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output, num_indices, embedding_dim);
}
