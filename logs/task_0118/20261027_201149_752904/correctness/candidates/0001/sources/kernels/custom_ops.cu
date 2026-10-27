#include <cuda_runtime.h>

// BatchNorm2d eval forward kernel
// x: input tensor [N, C, H, W] with arbitrary strides
// weight, bias, running_mean, running_var: size C
// eps: scalar
// output: same shape as x, contiguous
__global__ void batchnorm2d_eval_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ out,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if (idx >= total) return;

    // Decompose linear index into n,c,h,w using output strides (assumed contiguous)
    int n = idx / (out_stride_n);
    int rem = idx % out_stride_n;
    int c = rem / out_stride_c;
    rem = rem % out_stride_c;
    int h = rem / out_stride_h;
    int w = rem % out_stride_h;

    // Compute corresponding input index using input strides
    long long in_idx = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w;

    float val = x[in_idx];
    float scale = weight[c] / sqrtf(running_var[c] + eps);
    float shift = bias[c] - running_mean[c] * scale;
    out[idx] = val * scale + shift;
}

// Sum over dim=1 (channels) with keepdim=True
// input: [N, C, H, W] arbitrary strides, output: [N, 1, H, W] contiguous
__global__ void sum_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * H * W;
    if (idx >= total) return;

    // Output index corresponds to (n, 0, h, w) in output contiguous tensor
    int n = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        long long in_idx = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w;
        sum += x[in_idx];
    }
    out[idx] = sum;
}

// Launcher for batchnorm2d_eval_forward
extern "C" void batchnorm2d_eval_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* out,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    batchnorm2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, running_mean, running_var, eps, out,
        N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// Launcher for sum_dim_keepdim_forward
extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long x_stride_n, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * H * W;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    sum_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, out, N, C, H, W, x_stride_n, x_stride_c, x_stride_h, x_stride_w);
}
