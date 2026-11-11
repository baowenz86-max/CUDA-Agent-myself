#include <cuda_runtime.h>

// mean_dim1_keepdim: reduces over dim=1 (C) and keeps dimension
// Input: x of shape [N, C, H, W], arbitrary strides
// Output: y of shape [N, 1, H, W]
// Each thread computes one output element at (n, 0, h, w)
__global__ void mean_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c,
    long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_0,
    long long output_stride_h, long long output_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (H * W);

    const float* x_base = x + n * input_stride_n + h * input_stride_h + w * input_stride_w;
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += x_base[c * input_stride_c];
    }
    float mean = sum / static_cast<float>(C);

    float* y_ptr = y + n * output_stride_n + 0 * output_stride_0 + h * output_stride_h + w * output_stride_w;
    *y_ptr = mean;
}

// relu: element-wise max(0, x), same shape and strides as input
__global__ void relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = x[idx];
        y[idx] = val > 0.0f ? val : 0.0f;
    }
}

// Launcher for mean_dim1_keepdim, accepts raw pointers and tensor metadata.
// The binding layer will allocate output tensor and pass strides.
extern "C" void mean_dim1_keepdim_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c,
    long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_0,
    long long output_stride_h, long long output_stride_w,
    cudaStream_t stream)
{
    if (N <= 0 || H <= 0 || W <= 0) return;
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_0, output_stride_h, output_stride_w);
}

// Launcher for relu, just element-wise on flat arrays
extern "C" void relu_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(x, y, numel);
}
