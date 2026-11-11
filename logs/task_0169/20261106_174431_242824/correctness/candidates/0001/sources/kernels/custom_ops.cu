// CUDA kernels for batch_norm2d_forward, hardsigmoid_forward, add_bias_forward
// All launchers are extern "C" and accept raw pointers and cudaStream_t
#include <cuda_runtime.h>

// BatchNorm2d evaluation mode kernel
// x: input tensor shape [N,C,H,W] (float32)
// weight, bias, running_mean, running_var: per-channel arrays [C]
// eps: epsilon
// output: same shape as x
// The kernel computes y = weight * (x - running_mean) / sqrt(running_var + eps) + bias
__global__ void batch_norm2d_eval_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long total = (long long)N * C * H * W;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        // decode indices
        int w = idx % W;
        long long rem = idx / W;
        int h = rem % H;
        rem /= H;
        int c = rem % C;
        int n = rem / C;

        // compute input and output offsets using strides
        long long x_offset = (long long)n * x_stride_n + (long long)c * x_stride_c +
                             (long long)h * x_stride_h + (long long)w * x_stride_w;
        long long out_offset = (long long)n * out_stride_n + (long long)c * out_stride_c +
                               (long long)h * out_stride_h + (long long)w * out_stride_w;

        float mean = running_mean[c];
        float var = running_var[c];
        float invstd = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;

        float val = (x[x_offset] - mean) * invstd;
        output[out_offset] = gamma * val + beta;
    }
}

// Launcher for batch_norm2d_forward
// Assumes x and output may be non-contiguous, but weight, bias, running_mean, running_var are contiguous.
// Pass nullptr for weight/bias if not affine.
extern "C" void batch_norm2d_forward_launcher(
    const float* x,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    bool training,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    // We only implement evaluation mode as required by model_new.py (training=False)
    // If training is true, this launcher will not be used; binding should check.
    if (training) {
        // Not supported; do nothing or handle error via binding
        return;
    }

    long long total = (long long)N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Grid cap: limit to 65535 blocks (for compatibility), use grid-stride loop
    if (blocks > 65535) blocks = 65535;

    batch_norm2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        x, output, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// Hardsigmoid kernel: y = clamp(x * slope + 0.5, 0, 1)
__global__ void hardsigmoid_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    float slope,
    long long numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = x[idx] * slope + 0.5f;
        output[idx] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}

extern "C" void hardsigmoid_forward_launcher(
    const float* x,
    float* output,
    float slope,
    long long numel,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    // Grid-stride loop version to handle any size
    // Use separate kernel with grid-stride to avoid huge grid dimensions
    // We can reuse same kernel with idx < numel; if blocks limited, need grid-stride loop.
    // Let's implement a grid-stride version to be safe.
    // For simplicity, implement a separate grid-stride kernel or make the kernel grid-stride.
    // The above kernel is one-element-per-thread without grid-stride; if numel > 65535*256, multiple elements per thread needed.
    // To avoid complexity, we implement a grid-stride loop in the same kernel using a second function.
    // However, we can just set blocks to cover all elements up to max grid and rely on grid-stride loop inside.
    // We'll define a grid-stride version below.
}

// Grid-stride hardsigmoid kernel
__global__ void hardsigmoid_grid_stride_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    float slope,
    long long numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float v = x[i] * slope + 0.5f;
        output[i] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}

extern "C" void hardsigmoid_forward_launcher(
    const float* x,
    float* output,
    float slope,
    long long numel,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1; // in case numel==0?
    hardsigmoid_grid_stride_kernel<<<blocks, threads, 0, stream>>>(
        x, output, slope, numel);
}

// Add bias kernel: output = x + bias[c], where bias has shape [C,1,1]
// Bias is contiguous with C elements.
__global__ void add_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long total = (long long)N * C * H * W;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int w = idx % W;
        long long rem = idx / W;
        int h = rem % H;
        rem /= H;
        int c = rem % C;
        int n = rem / C;

        long long x_offset = (long long)n * x_stride_n + (long long)c * x_stride_c +
                             (long long)h * x_stride_h + (long long)w * x_stride_w;
        long long out_offset = (long long)n * out_stride_n + (long long)c * out_stride_c +
                               (long long)h * out_stride_h + (long long)w * out_stride_w;

        output[out_offset] = x[x_offset] + bias[c];
    }
}

extern "C" void add_bias_forward_launcher(
    const float* x,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(
        x, bias, output, N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
