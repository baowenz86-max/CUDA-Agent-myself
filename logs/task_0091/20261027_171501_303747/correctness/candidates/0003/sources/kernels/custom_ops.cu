#include <cuda_runtime.h>
#include <algorithm>

// -----------------------------------------------------------------------------
// triu kernel (grid-stride loop)
// -----------------------------------------------------------------------------
__global__ void triu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total,
    int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H; // now tmp is n*C + c
        // We don't need C to split further; h and w suffice for triu.
        float val = 0.0f;
        if (h <= w) {
            // Compute offset from n,c,h,w using strides.
            // Since idx = ((n*C + c)*H + h)*W + w,
            // n*C + c = tmp, so n = tmp / C and c = tmp % C.
            // We need C, but we can pass it or derive? We'll pass C to avoid extra division.
            // To keep this kernel simple, we'll pass C as parameter.
            // We'll include C in the signature for clarity.
        }
    }
}

// Final triu kernel that includes C parameter for decoding.
__global__ void triu_kernel_with_C(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total,
    int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        float val = 0.0f;
        if (h <= w) {
            val = input[n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w];
        }
        output[n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w] = val;
    }
}

extern "C" void triu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total <= 0) return;
    const int threads = 256;
    int grid = (int)std::min<long long>(65535, (total + threads - 1) / threads);
    if (grid == 0) grid = 1;
    triu_kernel_with_C<<<grid, threads, 0, stream>>>(
        output, input, total, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// -----------------------------------------------------------------------------
// add_bias kernel
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    long long total,
    int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        float b = bias[c];
        float val = input[n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w] + b;
        output[n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w] = val;
    }
}

extern "C" void add_bias_forward_launcher(
    float* output,
    const float* input,
    const float* bias,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total <= 0) return;
    const int threads = 256;
    int grid = (int)std::min<long long>(65535, (total + threads - 1) / threads);
    if (grid == 0) grid = 1;
    add_bias_kernel<<<grid, threads, 0, stream>>>(
        output, input, bias, total, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// -----------------------------------------------------------------------------
// conv2d forward kernel (direct convolution, groups=1 only)
// -----------------------------------------------------------------------------
__global__ void conv2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // may be nullptr
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int stride, int padding, int dilation,
    int groups)
{
    // Each thread computes one output element.
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    long long tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    // Groups must be 1.
    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < kH; ++kh) {
            int h_in = h_out * stride - padding + kh * dilation;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int w_in = w_out * stride - padding + kw * dilation;
                if (w_in < 0 || w_in >= W_in) continue;
                float input_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                float weight_val = weight[((c_out * C_in + c_in) * kH + kh) * kW + kw];
                acc += input_val * weight_val;
            }
        }
    }
    if (bias != nullptr) {
        acc += bias[c_out];
    }
    output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = acc;
}

extern "C" void conv2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int stride, int padding, int dilation, int groups,
    cudaStream_t stream)
{
    long long total = (long long)N * C_out * H_out * W_out;
    if (total <= 0) return;
    const int threads = 256;
    int grid = (int)std::min<long long>(65535, (total + threads - 1) / threads);
    if (grid == 0) grid = 1;
    conv2d_kernel<<<grid, threads, 0, stream>>>(
        output, input, weight, bias,
        N, C_in, H_in, W_in, C_out, kH, kW, H_out, W_out,
        stride, padding, dilation, groups);
}
