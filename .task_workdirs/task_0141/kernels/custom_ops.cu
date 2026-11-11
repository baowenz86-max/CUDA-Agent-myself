#include <cuda_runtime.h>

// ============================================================================
// Leaky ReLU kernel
// ============================================================================
template<int BLOCK_SIZE>
__global__ void leaky_relu_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    long long numel,
    int ndim,
    const long long* __restrict__ shape,
    const long long* __restrict__ in_strides,
    const long long* __restrict__ out_strides,
    float negative_slope)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        long long rem = idx;
        long long in_offset = 0;
        long long out_offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            long long coord = rem % shape[d];
            rem /= shape[d];
            in_offset += coord * in_strides[d];
            out_offset += coord * out_strides[d];
        }
        float val = in[in_offset];
        out[out_offset] = (val >= 0.0f) ? val : negative_slope * val;
    }
}

// ============================================================================
// MaxPool2d kernel (kernel_size=2, stride=2, no padding, dilation=1)
// ============================================================================
__global__ void max_pool2d_2x2_stride2_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    int outH, int outW,
    const long long* __restrict__ in_strides,
    const long long* __restrict__ out_strides)
{
    long long total_out = (long long)N * C * outH * outW;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_out; idx += stride) {
        int ow = idx % outW;
        long long tmp = idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        int ih = oh * 2;
        int iw = ow * 2;

        long long in_base = (long long)n * in_strides[0] + (long long)c * in_strides[1]
                         + (long long)ih * in_strides[2] + (long long)iw * in_strides[3];

        float v0 = in[in_base];
        float v1 = in[in_base + in_strides[3]];
        float v2 = in[in_base + in_strides[2]];
        float v3 = in[in_base + in_strides[2] + in_strides[3]];

        float max_val = v0;
        if (v1 > max_val) max_val = v1;
        if (v2 > max_val) max_val = v2;
        if (v3 > max_val) max_val = v3;

        long long out_offset = (long long)n * out_strides[0] + (long long)c * out_strides[1]
                            + (long long)oh * out_strides[2] + (long long)ow * out_strides[3];
        out[out_offset] = max_val;
    }
}

// ============================================================================
// Conv2d kernel (NCHW, stride=1, dilation=1, no padding)
// ============================================================================
// Supports arbitrary kernel size R x S, input C, output K.
// Each thread computes one output element.
__global__ void conv2d_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be null
    int N, int C, int H, int W,
    int K, int R, int S,
    int outH, int outW,
    int use_bias)
{
    long long total_out = (long long)N * K * outH * outW;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_out; idx += stride) {
        int ow = idx % outW;
        long long tmp = idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int k = tmp % K;
        int n = tmp / K;

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                int ih = oh + r;  // stride=1, dilation=1, pad=0
                for (int s = 0; s < S; ++s) {
                    int iw = ow + s;
                    float input_val = input[((long long)n * C + c) * H * W + (long long)ih * W + iw];
                    float weight_val = weight[((long long)k * C + c) * R * S + (long long)r * S + s];
                    sum += input_val * weight_val;
                }
            }
        }
        if (use_bias) {
            sum += bias[k];
        }
        out[((long long)n * K + k) * outH * outW + (long long)oh * outW + ow] = sum;
    }
}

// ============================================================================
// Launcher functions (extern "C" for binding layer)
// ============================================================================

extern "C" void leaky_relu_launcher(
    float* out,
    const float* in,
    long long numel,
    int ndim,
    const long long* shape,
    const long long* in_strides,
    const long long* out_strides,
    float negative_slope,
    cudaStream_t stream)
{
    const int block_size = 256;
    long long blocks = (numel + block_size - 1) / block_size;
    if (blocks > 0) {
        leaky_relu_kernel<256><<<(unsigned int)blocks, block_size, 0, stream>>>(
            out, in, numel, ndim, shape, in_strides, out_strides, negative_slope);
    }
}

extern "C" void max_pool2d_2x2_stride2_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int outH, int outW,
    const long long* in_strides,
    const long long* out_strides,
    cudaStream_t stream)
{
    long long total_out = (long long)N * C * outH * outW;
    const int block_size = 256;
    long long blocks = (total_out + block_size - 1) / block_size;
    if (blocks > 0) {
        max_pool2d_2x2_stride2_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
            out, in, N, C, H, W, outH, outW, in_strides, out_strides);
    }
}

extern "C" void conv2d_launcher(
    float* out,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    int K, int R, int S,
    int /*stride_h*/, int /*stride_w*/,
    int /*pad_h*/, int /*pad_w*/,
    int /*dilation_h*/, int /*dilation_w*/,
    int use_bias,
    cudaStream_t stream)
{
    // Hardcode stride=1, dilation=1, padding=0 as required by task contract
    int outH = (H - R) + 1;
    int outW = (W - S) + 1;
    long long total_out = (long long)N * K * outH * outW;
    const int block_size = 256;
    long long blocks = (total_out + block_size - 1) / block_size;
    if (blocks > 0) {
        conv2d_forward_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
            out, input, weight, bias, N, C, H, W, K, R, S, outH, outW, use_bias);
    }
}
