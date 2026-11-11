#include <cuda_runtime.h>
#include <cmath>

// Elementwise tanh kernel. Input can be non-contiguous, using per-dimension strides.
// We linearize thread index to flat offset over the 4D tensor (N, C, H, W) and
// compute coordinates via division/modulo.
__global__ void tanh_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long in_stride_n,  long long in_stride_c,  long long in_stride_h,  long long in_stride_w,
    long long total_elements)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long linear = idx; linear < total_elements; linear += stride) {
        // decompose linear index
        int w = linear % W;
        long long rem = linear / W;
        int h = rem % H;
        rem /= H;
        int c = rem % C;
        int n = rem / C;

        long long out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        long long in_offset  = n * in_stride_n  + c * in_stride_c  + h * in_stride_h  + w * in_stride_w;
        out[out_offset] = tanhf(in[in_offset]);
    }
}

// BatchNorm2d evaluation mode kernel. Each thread processes one (n,c,h,w) element.
// We use grid-stride loop over total elements. For each element, we compute channel index c
// and apply scale and shift from precomputed arrays. Weight, bias, running_mean, running_var
// are assumed contiguous float arrays of length C (binding layer will pass contiguous).
__global__ void batch_norm2d_eval_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long in_stride_n,  long long in_stride_c,  long long in_stride_h,  long long in_stride_w,
    long long total_elements)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long linear = idx; linear < total_elements; linear += stride) {
        int w = linear % W;
        long long rem = linear / W;
        int h = rem % H;
        rem /= H;
        int c = rem % C;
        int n = rem / C;

        long long out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        long long in_offset  = n * in_stride_n  + c * in_stride_c  + h * in_stride_h  + w * in_stride_w;

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float scale = weight[c] * inv_std;
        float shift = bias[c] - mean * scale;
        out[out_offset] = in[in_offset] * scale + shift;
    }
}

extern "C" void tanh_forward_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long in_stride_n,  long long in_stride_c,  long long in_stride_h,  long long in_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total == 0) return;
    const int block = 256;
    long long grid_ll = (total + block - 1) / block;
    // Cap grid to avoid exceeding max grid dim; grid-stride loop handles rest
    int grid = (int)min(grid_ll, (long long)65535 * 32); // conservative: 65535*32 ~2.1M blocks > needed
    tanh_kernel<<<grid, block, 0, stream>>>(
        out, in, N, C, H, W,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        in_stride_n,  in_stride_c,  in_stride_h,  in_stride_w,
        total);
}

extern "C" void batch_norm2d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long in_stride_n,  long long in_stride_c,  long long in_stride_h,  long long in_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total == 0) return;
    const int block = 256;
    long long grid_ll = (total + block - 1) / block;
    int grid = (int)min(grid_ll, (long long)65535 * 32);
    batch_norm2d_eval_kernel<<<grid, block, 0, stream>>>(
        out, in, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        in_stride_n,  in_stride_c,  in_stride_h,  in_stride_w,
        total);
}
