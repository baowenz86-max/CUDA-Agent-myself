// custom_ops.cu
// Contains non-convolution CUDA kernels and launchers.
// Conv/ConvTranspose will use cuDNN implemented in C++ binding later.

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

    // Number of elements per channel = batch * out_width
    long long n = (long long)batch * out_width;

    // Each thread accumulates over multiple elements via grid-stride within block
    float sum = 0.0f;
    float sumsq = 0.0f;

    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        // Linear index within channel's collection (b, w)
        int w = i % out_width;
        int b = i / out_width;
        long long in_idx = (long long)b * input_batch_stride +
                           (long long)c * input_channel_stride +
                           (long long)w * input_width_stride;
        float val = input[in_idx];
        sum += val;
        sumsq += val * val;
    }

    // Store partial sums to shared memory
    s_sum[threadIdx.x] = sum;
    s_sq[threadIdx.x] = sumsq;
    __syncthreads();

    // Reduce within block
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
        // Numerical stability: if var slightly negative due to floating point, clamp to 0
        if (var < 0.0f) var = 0.0f;
        // unbiased standard deviation: divide by (n-1)
        float denom = (float)(n - 1);
        float std_val = (denom > 0.0f) ? sqrtf(var * (float)n / denom) : 0.0f;
        output[c] = std_val;
    }
}

// ---------------------------------------------------------------------------
// Launchers (extern "C" for binding layer)
// ---------------------------------------------------------------------------

// bias_add_broadcast forward
// input, bias, output are device pointers. bias is [out_channels] contiguous.
// Input is arbitrary strides. Output is contiguous [batch, out_channels, out_width].
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
    // Cap grid to avoid exceeding limits (though for typical sizes it won't be huge)
    int grid = (int)min(grid_ll, (long long)1073741823);
    if (grid <= 0) return;
    bias_add_broadcast_kernel<<<grid, block, 0, stream>>>(
        input, bias, output,
        batch, out_channels, out_width,
        input_batch_stride, input_channel_stride, input_width_stride);
}

// std_batch_width_unbiased forward
// input arbitrary strides, output [out_channels]
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
    int grid = out_channels;  // one block per channel
    if (grid <= 0) return;
    size_t shared_mem = 2 * block * sizeof(float);
    std_batch_width_unbiased_kernel<<<grid, block, shared_mem, stream>>>(
        input, output,
        batch, out_channels, out_width,
        input_batch_stride, input_channel_stride, input_width_stride);
}
