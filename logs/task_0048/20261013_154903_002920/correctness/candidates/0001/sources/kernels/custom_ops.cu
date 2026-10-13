#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Adaptive average pooling over last two dimensions (height, width)
// Input: x shape [batch, channels, depth, height, width]
// Output: y shape [batch, channels, depth, 1, 1]
// Each output element is the mean of all elements in the corresponding
// (batch, channel, depth) slice over height and width.
// -----------------------------------------------------------------------------
extern "C" __global__ void adaptive_avg_pool2d_5d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long total_output_elements)  // batch * channels * depth
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total_output_elements; i += stride) {
        // i = ((b * channels) + c) * depth + d
        int d = i % depth;
        long long tmp = i / depth;
        int c = tmp % channels;
        int b = tmp / channels;

        long long base = ((long long)b * channels + c) * depth * height * width
                         + (long long)d * height * width;
        float sum = 0.0f;
        for (int hw = 0; hw < height * width; ++hw) {
            sum += x[base + hw];
        }
        y[i] = sum / (float)(height * width);
    }
}

extern "C" void adaptive_avg_pool2d_5d_launcher(
    const float* x,
    float* y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream)
{
    long long total = (long long)batch * channels * depth;
    if (total <= 0) return;

    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    adaptive_avg_pool2d_5d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, batch, channels, depth, height, width, total);
}

// -----------------------------------------------------------------------------
// LeakyReLU elementwise: y = x >= 0 ? x : x * negative_slope
// Input and output have the same shape and layout (contiguous assumed by caller)
// -----------------------------------------------------------------------------
extern "C" __global__ void leaky_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long n,
    float negative_slope)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < n; i += stride) {
        float val = x[i];
        y[i] = val >= 0.0f ? val : val * negative_slope;
    }
}

extern "C" void leaky_relu_launcher(
    const float* x,
    float* y,
    long long n,
    float negative_slope,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int threads = 256;
    long long blocks_ll = (n + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(x, y, n, negative_slope);
}

// -----------------------------------------------------------------------------
// Add bias with shape [channels, 1, 1, 1] broadcast over [batch, channels, depth, height, width]
// For each element, add bias[channel]
// -----------------------------------------------------------------------------
extern "C" __global__ void add_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    long long total,       // total elements = batch * channels * depth * height * width
    int channels,
    int inner_size)        // depth * height * width
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        // i = (b * channels + c) * inner_size + offset
        int c = (i / inner_size) % channels;
        y[i] = x[i] + bias[c];
    }
}

extern "C" void add_bias_launcher(
    const float* x,
    const float* bias,
    float* y,
    long long total,
    int channels,
    int inner_size,
    cudaStream_t stream)
{
    if (total <= 0) return;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    add_bias_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, total, channels, inner_size);
}

// -----------------------------------------------------------------------------
// Clamp elementwise: y = min(max(x, min_val), max_val)
// -----------------------------------------------------------------------------
extern "C" __global__ void clamp_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long n,
    float min_val,
    float max_val)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < n; i += stride) {
        float val = x[i];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        y[i] = val;
    }
}

extern "C" void clamp_launcher(
    const float* x,
    float* y,
    long long n,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int threads = 256;
    long long blocks_ll = (n + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    clamp_kernel<<<blocks, threads, 0, stream>>>(x, y, n, min_val, max_val);
}
