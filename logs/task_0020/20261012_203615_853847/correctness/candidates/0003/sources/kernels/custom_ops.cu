#include <cuda_runtime.h>

// CUDA kernel for torch.sign
extern "C" __global__ void sign_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    }
}

// Launcher for sign operation
extern "C" void sign_forward(const float* input, float* output, int numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sign_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// CUDA kernel for conv_transpose1d (simplified for specific case: in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
// Input: (batch, in_channels, in_length)
// Weight: (in_channels, out_channels, kernel_size)
// Bias: (out_channels)
// Output: (batch, out_channels, out_length) where out_length = (in_length-1)*stride - 2*padding + kernel_size + output_padding
extern "C" __global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * ((in_length - 1) * stride - 2 * padding + kernel_size + output_padding);
    if (idx >= total_elements) return;

    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_pos = idx;
    int b = out_pos / (out_channels * out_length);
    int rem = out_pos % (out_channels * out_length);
    int oc = rem / out_length;
    int ol = rem % out_length;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int il = ol + padding - k;
            if (il >= 0 && il < in_length && il % stride == 0) {
                int input_index = ((b * in_channels + ic) * in_length + (il / stride));
                int weight_index = (ic * out_channels + oc) * kernel_size + k;
                sum += input[input_index] * weight[weight_index];
            }
        }
    }
    output[idx] = sum + bias[oc];
}

// Launcher for conv_transpose1d
extern "C" void conv_transpose1d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream
) {
    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    int total_elements = batch * out_channels * out_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, in_length,
        kernel_size, stride, padding, output_padding
    );
}

// CUDA kernel for embedding lookup
// indices: (batch, length) int64
// weight: (num_embeddings, embedding_dim) float32
// output: (batch, length, embedding_dim) float32
extern "C" __global__ void embedding_kernel(
    const long long* indices,
    const float* weight,
    float* output,
    int batch,
    int length,
    int embedding_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * length * embedding_dim;
    if (idx >= total_elements) return;

    int b = idx / (length * embedding_dim);
    int rem = idx % (length * embedding_dim);
    int l = rem / embedding_dim;
    int d = rem % embedding_dim;

    long long index = indices[b * length + l];
    // Assume indices are valid
    output[idx] = weight[index * embedding_dim + d];
}

// Launcher for embedding
extern "C" void embedding_forward(
    const long long* indices,
    const float* weight,
    float* output,
    int batch,
    int length,
    int embedding_dim,
    cudaStream_t stream
) {
    int total_elements = batch * length * embedding_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output, batch, length, embedding_dim
    );
}
