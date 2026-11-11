// custom_ops.cu
// Contains CUDA kernels and launchers for all custom ops.

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// bias_add_broadcast kernel
// input: [batch, out_channels, out_width] (supports arbitrary strides)
// bias:  [out_channels] (contiguous)
// output: [batch, out_channels, out_width] (contiguous)
// ---------------------------------------------------------------------------
__global__ void bias_add_broadcast_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int out_channels,
    int out_width,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_width_stride)
{
    long long total = (long long)batch * out_channels * out_width;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % out_width;
        long long tmp = idx / out_width;
        int c = tmp % out_channels;
        int b = tmp / out_channels;

        long long in_idx = (long long)b * input_batch_stride +
                           (long long)c * input_channel_stride +
                           (long long)w * input_width_stride;

        output[idx] = input[in_idx] + bias[c];
    }
}

// ---------------------------------------------------------------------------
// std_batch_width_unbiased kernel
// input: [batch, out_channels, out_width] (arbitrary strides)
// output: [out_channels]
// Computes unbiased standard deviation over dims [0, 2] for each channel.
// Each block handles one channel.
// ---------------------------------------------------------------------------
__global__ void std_batch_width_unbiased_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int out_channels,
    int out_width,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_width_stride)
{
    extern __shared__ float sdata[]; // size = 2 * blockDim.x
    float* s_sum = sdata;
    float* s_sq  = sdata + blockDim.x;

    int c = blockIdx.x;
    if (c >= out_channels) return;

    long long n = (long long)batch * out_width;

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        int w = i % out_width;
        int b = i / out_width;
        long long in_idx = (long long)b * input_batch_stride +
                           (long long)c * input_channel_stride +
                           (long long)w * input_width_stride;
        float val = input[in_idx];
        sum += val;
        sumsq += val * val;
    }

    s_sum[threadIdx.x] = sum;
    s_sq[threadIdx.x] = sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_sum = s_sum[0];
        float total_sq = s_sq[0];
        float mean = total_sum / (float)n;
        float var = (total_sq / (float)n) - mean * mean;
        if (var < 0.0f) var = 0.0f;
        float denom = (float)(n - 1);
        float std_val = (denom > 0.0f) ? sqrtf(var * (float)n / denom) : 0.0f;
        output[c] = std_val;
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose1d forward kernel (single block per output element)
// input: [batch, in_channels, in_width]
// weight: [in_channels, out_channels, kernel_size]
// bias: [out_channels] (optional, may be nullptr)
// output: [batch, out_channels, out_width]
// stride, padding, output_padding are integers
// ---------------------------------------------------------------------------
__global__ void conv_transpose1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_width,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding)
{
    long long total = (long long)batch * out_channels * out_width;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride_total = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride_total) {
        int w = idx % out_width;
        long long tmp = idx / out_width;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        float val = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                // input index: for transposed conv, output position w corresponds to input position i such that
                // w = i * stride - padding + k + output_padding? Actually output_padding is just extra zeros at the end.
                // Standard transposed conv: output[w] += input[i] * weight[ic, oc, k] if w == i*stride - padding + k
                // Rearrange: i*stride = w + padding - k
                int numerator = w + padding - k;
                if (numerator >= 0 && numerator % stride == 0) {
                    int i = numerator / stride;
                    if (i >= 0 && i < in_width) {
                        // input index: [b, ic, i]
                        long long in_idx = ((long long)b * in_channels + ic) * in_width + i;
                        // weight index: [ic, oc, k]
                        long long w_idx = ((long long)ic * out_channels + oc) * kernel_size + k;
                        val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            val += bias[oc];
        }
        // For positions beyond the valid conv result (due to output_padding), value should be 0,
        // but our loop already yields 0 if no valid contributions, so val remains 0 (plus bias).
        // Note: PyTorch adds bias to all positions, so bias should be added even in output_padding region.
        // That matches our unconditional bias addition.
        output[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Launchers (extern "C" for binding layer)
// ---------------------------------------------------------------------------

// bias_add_broadcast forward
// input, bias, output are device pointers. bias is [out_channels] contiguous.
extern "C" void bias_add_broadcast_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int out_channels,
    int out_width,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_width_stride,
    cudaStream_t stream)
{
    long long total = (long long)batch * out_channels * out_width;
    int block = 256;
    long long grid_ll = (total + block - 1) / block;
    int grid = (int)min(grid_ll, (long long)1073741823);
    if (grid <= 0) return;
    bias_add_broadcast_kernel<<<grid, block, 0, stream>>>(
        input, bias, output,
        batch, out_channels, out_width,
        input_batch_stride, input_channel_stride, input_width_stride);
}

// std_batch_width_unbiased forward
extern "C" void std_batch_width_unbiased_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int out_channels,
    int out_width,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_width_stride,
    cudaStream_t stream)
{
    int block = 256;
    int grid = out_channels;
    if (grid <= 0) return;
    size_t shared_mem = 2 * block * sizeof(float);
    std_batch_width_unbiased_kernel<<<grid, block, shared_mem, stream>>>(
        input, output,
        batch, out_channels, out_width,
        input_batch_stride, input_channel_stride, input_width_stride);
}

// conv_transpose1d forward
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_width,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    long long total = (long long)batch * out_channels * out_width;
    int block = 256;
    long long grid_ll = (total + block - 1) / block;
    int grid = (int)min(grid_ll, (long long)1073741823);
    if (grid <= 0) return;
    conv_transpose1d_forward_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, in_width, out_width,
        kernel_size, stride, padding, output_padding);
}
