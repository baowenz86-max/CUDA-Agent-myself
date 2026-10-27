#include <cuda_runtime.h>

// ConvTranspose1d kernel: each thread computes one output element.
// Output shape: [batch, out_channels, length_out]
// Input shape: [batch, in_channels, length_in]
// Weight shape: [in_channels, out_channels, kernel_size]
// Bias shape: [out_channels] (optional, nullptr if absent)
__global__ void conv_transpose1d_kernel(
    float* __restrict__ output,
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
    int output_padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * length_out) return;

    int b = idx / (out_channels * length_out);
    int rem = idx % (out_channels * length_out);
    int oc = rem / length_out;
    int ol = rem % length_out;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int il = ol + padding - k;
            if (il % stride != 0) continue;
            il /= stride;
            if (il < 0 || il >= length_in) continue;
            sum += input[(b * in_channels + ic) * length_in + il] *
                   weight[(ic * out_channels + oc) * kernel_size + k];
        }
    }
    if (bias != nullptr) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

// Softmax over dim=1 for a 3D tensor of shape [batch, channels, length].
// Each thread block handles one (batch, length) position, computing softmax over channels.
template <int BLOCK_SIZE>
__global__ void softmax_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int channels,
    int length)
{
    int idx = blockIdx.x;
    int b = idx / length;
    int l = idx % length;

    const float* in_col = input + (size_t)b * channels * length + l;
    float* out_col = output + (size_t)b * channels * length + l;

    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int stride = BLOCK_SIZE;

    float max_val = -INFINITY;
    for (int c = tid; c < channels; c += stride) {
        float v = in_col[(size_t)c * length];
        if (v > max_val) max_val = v;
    }

    smem[tid] = max_val;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }
    float block_max = smem[0];
    __syncthreads();

    float sum = 0.0f;
    for (int c = tid; c < channels; c += stride) {
        float v = in_col[(size_t)c * length];
        float e = expf(v - block_max);
        sum += e;
        out_col[(size_t)c * length] = e;
    }

    smem[tid] = sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float total_sum = smem[0];
    __syncthreads();

    for (int c = tid; c < channels; c += stride) {
        out_col[(size_t)c * length] /= total_sum;
    }
}

// Launcher for conv_transpose1d_forward
extern "C" void conv_transpose1d_forward_launcher(
    float* output,
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
    cudaStream_t stream)
{
    int total_elements = batch * out_channels * length_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias,
        batch, in_channels, out_channels, length_in, length_out, kernel_size,
        stride, padding, output_padding);
}

// Launcher for softmax_dim1_forward
extern "C" void softmax_dim1_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int total_positions = batch * length;
    dim3 grid(total_positions);
    dim3 block(BLOCK_SIZE);
    softmax_dim1_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        output, input, batch, channels, length);
}
