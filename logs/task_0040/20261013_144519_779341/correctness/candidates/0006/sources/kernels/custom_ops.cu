#include <cuda_runtime.h>

// =============================================================================
// add_bias_broadcast kernel
// Adds bias of shape [out_channels, 1] to input of shape [batch, out_channels, Lout]
// Broadcasting along batch and length dimensions.
// =============================================================================
__global__ void add_bias_broadcast_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int batch,
    int out_channels,
    int Lout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * Lout;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // decompose linear index into (b, c, l)
        int l = i % Lout;
        int temp = i / Lout;
        int c = temp % out_channels;
        int b = temp / out_channels;

        output[i] = input[i] + bias[c];
    }
}

extern "C" void add_bias_broadcast_launcher(
    float* output,
    const float* input,
    const float* bias,
    int batch,
    int out_channels,
    int Lout,
    cudaStream_t stream)
{
    int total = batch * out_channels * Lout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    add_bias_broadcast_kernel<<<blocks, threads, 0, stream>>>(
        output, input, bias, batch, out_channels, Lout);
}

// =============================================================================
// roll_dim2 kernel
// Circular shift along last dimension: output[b,c,k] = input[b,c, (k - shift) mod Lout]
// for positive shift.
// =============================================================================
__global__ void roll_dim2_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int out_channels,
    int Lout,
    int shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * Lout;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int k = i % Lout;
        int temp = i / Lout;
        int c = temp % out_channels;
        int b = temp / out_channels;

        // compute source index
        int src_k = (k - shift) % Lout;
        if (src_k < 0) src_k += Lout;
        int src_idx = b * (out_channels * Lout) + c * Lout + src_k;
        output[i] = input[src_idx];
    }
}

extern "C" void roll_dim2_launcher(
    float* output,
    const float* input,
    int batch,
    int out_channels,
    int Lout,
    int shift,
    cudaStream_t stream)
{
    int total = batch * out_channels * Lout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    roll_dim2_kernel<<<blocks, threads, 0, stream>>>(
        output, input, batch, out_channels, Lout, shift);
}

// =============================================================================
// view_flatten kernel
// Flattening is just a contiguous copy, since input is already contiguous and
// output is contiguous with the same number of elements. The binding will
// allocate an output tensor with shape [batch, out_channels*Lout] and pass
// raw pointers; this kernel simply copies the data.
// =============================================================================
__global__ void view_flatten_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void view_flatten_launcher(
    float* output,
    const float* input,
    int total,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    view_flatten_kernel<<<blocks, threads, 0, stream>>>(output, input, total);
}
