#include <cuda_runtime.h>
#include <cstdint>

// Generic 3D max pooling kernel.
// Each thread computes one output element (n,c,od,oh,ow).
// Input tensor is assumed to have strides [sN, sC, sD, sH, sW] (in elements).
__global__ void max_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    long long sN, long long sC, long long sD, long long sH, long long sW,
    int kD, int kH, int kW,
    int sD_stride, int sH_stride, int sW_stride,
    int pD, int pH, int pW,
    int dilation,
    int outD, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outD * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int tmp = idx / outW;
    int oh = tmp % outH;
    tmp /= outH;
    int od = tmp % outD;
    tmp /= outD;
    int c = tmp % C;
    int n = tmp / C;

    float max_val = -3.402823466e+38f;

    int d_start = od * sD_stride - pD;
    int h_start = oh * sH_stride - pH;
    int w_start = ow * sW_stride - pW;

    for (int kd = 0; kd < kD; ++kd) {
        int in_d = d_start + kd * dilation;
        if (in_d < 0 || in_d >= D) continue;
        for (int kh = 0; kh < kH; ++kh) {
            int in_h = h_start + kh * dilation;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int in_w = w_start + kw * dilation;
                if (in_w < 0 || in_w >= W) continue;
                long long offset = (long long)n * sN + (long long)c * sC +
                                   (long long)in_d * sD + (long long)in_h * sH + (long long)in_w * sW;
                float val = input[offset];
                if (val > max_val) max_val = val;
            }
        }
    }

    output[idx] = max_val;
}

// Host launcher that computes output dimensions and launches the kernel.
extern "C" void max_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    const long long* input_strides,
    int kD, int kH, int kW,
    int sD_stride, int sH_stride, int sW_stride,
    int pD, int pH, int pW,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream)
{
    auto compute_out = [&](int in, int k, int s, int p, int d) -> int {
        int numerator = in + 2 * p - d * (k - 1) - 1;
        if (ceil_mode) {
            return (numerator + s - 1) / s + 1;
        } else {
            return numerator / s + 1;
        }
    };

    int outD = compute_out(D, kD, sD_stride, pD, dilation);
    int outH = compute_out(H, kH, sH_stride, pH, dilation);
    int outW = compute_out(W, kW, sW_stride, pW, dilation);

    long long sN = input_strides[0];
    long long sC = input_strides[1];
    long long sD = input_strides[2];
    long long sH = input_strides[3];
    long long sW = input_strides[4];

    int total = N * C * outD * outH * outW;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    max_pool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        sN, sC, sD, sH, sW,
        kD, kH, kW,
        sD_stride, sH_stride, sW_stride,
        pD, pH, pW, dilation,
        outD, outH, outW);
}
