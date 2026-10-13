// Custom CUDA kernels for non-convolution operations.
// Conv2d is implemented separately using cuDNN in a C++ binding file.
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// add_bias_forward: out[n,c,h,w] = in[n,c,h,w] + bias[c]
// input may have arbitrary strides, so we use a generic 4D indexing kernel.
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    long long tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;

    long long in_idx = n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
    long long out_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
    out[out_idx] = in[in_idx] + bias[c];
}

// -----------------------------------------------------------------------------
// batch_norm_eval_forward: y = (x - mean) / sqrt(var + eps) * weight + bias
// -----------------------------------------------------------------------------
__global__ void batch_norm_eval_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    long long tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;

    long long in_idx = n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
    long long out_idx = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    float gamma = (weight != nullptr) ? weight[c] : 1.0f;
    float beta = (bias != nullptr) ? bias[c] : 0.0f;
    out[out_idx] = (in[in_idx] - mean) * inv_std * gamma + beta;
}

// -----------------------------------------------------------------------------
// avg_pool2d_forward: kernel_size=3, stride=1, padding=1
// Padding is zeros, count_include_pad=False (average over valid elements only).
// -----------------------------------------------------------------------------
__global__ void avg_pool2d_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H_out * W_out;
    if (idx >= total) return;

    int ow = idx % W_out;
    long long tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) continue;
            long long in_idx = n * in_stride_n + c * in_stride_c + ih * in_stride_h + iw * in_stride_w;
            sum += in[in_idx];
            ++count;
        }
    }
    long long out_idx = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w;
    out[out_idx] = sum / (float)count;
}

// -----------------------------------------------------------------------------
// sigmoid_forward: y = 1 / (1 + exp(-x))
// -----------------------------------------------------------------------------
__global__ void sigmoid_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    long long total)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C" so they can be called from binding .cpp)
// -----------------------------------------------------------------------------
extern "C" void add_bias_launcher(
    float* out,
    const float* in,
    const float* bias,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        out, in, bias, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling: binding layer may check, but we ignore here to keep simple.
    }
}

extern "C" void batch_norm_eval_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    batch_norm_eval_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        out, in, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

extern "C" void avg_pool2d_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H_out * W_out;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        out, in, N, C, H, W, H_out, W_out, kernel_size, stride, padding,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

extern "C" void sigmoid_launcher(
    float* out,
    const float* in,
    long long total,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    sigmoid_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(out, in, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}
