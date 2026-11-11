#include <cuda_runtime.h>
#include <cstdint>

// Conv2d forward kernel: input (N,C,H,W), weight (K,C,KH,KW), output (N,K,OH,OW)
// Supports stride_h=1, stride_w=1, padding_h=0, padding_w=0 as required by contract
extern "C" __global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int KH, int KW,
    int OH, int OW,
    int stride_h, int stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int k = (idx / (OW * OH)) % K;
    int n = idx / (OW * OH * K);

    float sum = bias != nullptr ? bias[k] : 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride_h + kh;  // padding_h=0
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride_w + kw;  // padding_w=0
                if (iw < 0 || iw >= W) continue;
                int input_idx = ((n * C + c) * H + ih) * W + iw;
                int weight_idx = ((k * C + c) * KH + kh) * KW + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[idx] = sum;
}

// ReLU kernel
extern "C" __global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// MaxPool2d kernel: input (N,C,H,W), output (N,C,OH,OW)
// Supports kernel_size, stride, padding (dilation=1)
extern "C" __global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OH, int OW,
    int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int c = (idx / (OW * OH)) % C;
    int n = idx / (OW * OH * C);

    float max_val = -1e30f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) continue;
            int input_idx = ((n * C + c) * H + ih) * W + iw;
            max_val = fmaxf(max_val, input[input_idx]);
        }
    }
    output[idx] = max_val;
}

// Launcher functions
extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int KH, int KW,
    int OH, int OW,
    int stride_h, int stride_w,
    cudaStream_t stream
) {
    int total = N * K * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, KH, KW, OH, OW, stride_h, stride_w
    );
}

extern "C" void relu_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}

extern "C" void maxpool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int OH, int OW,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
) {
    int total = N * C * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, OH, OW, kernel_size, stride, padding
    );
}
