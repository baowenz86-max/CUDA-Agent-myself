#include <cuda_runtime.h>
#include <math.h>

// Softmax over channel dimension (dim=1) for 4D tensor (N, C, H, W)
__global__ void softmax_channel_only_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w)
{
    long long total = (long long)N * H * W;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long idx = tid; idx < total; idx += stride) {
        int n = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;

        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            long long offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
            float val = x[offset];
            if (val > max_val) max_val = val;
        }

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            long long offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
            sum += expf(x[offset] - max_val);
        }

        for (int c = 0; c < C; ++c) {
            long long offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
            y[offset] = expf(x[offset] - max_val) / sum;
        }
    }
}

// PReLU activation: y = x if x > 0 else weight[c] * x
// c is channel index inferred from linear index and dimensions
__global__ void prelu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w)
{
    long long total = (long long)N * C * H * W;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = tid; i < total; i += stride) {
        // Compute channel index c for this linear index i
        // For contiguous NCHW layout: c = (i / (H*W)) % C
        // But we use strides to support non-contiguous tensors (stride_c may differ)
        // Derive n, c, h, w from linear index using strides is complex;
        // Instead we compute c by dividing by (H*W) modulo C, assuming NCHW layout
        // However, this assumes the linear index corresponds to NCHW ordering.
        // A safer approach for arbitrary strides: compute n = i / (C*H*W), then within a sample,
        // offset = i % (C*H*W); c = offset / (H*W); hw = offset % (H*W); h = hw / W; w = hw % W.
        // This works if the tensor is contiguous or at least the strides follow standard NCHW ordering.
        // Since we are given strides, we can compute exact offset using n,c,h,w if we can recover them.
        // Here we use the simple assumption that the linear index corresponds to contiguous NCHW layout,
        // but this may not be correct if strides are non-standard.
        // A more general approach: compute c by scanning possible n, h, w would be inefficient.
        // For this task, input x is contiguous NCHW, so this is acceptable.
        long long n = i / ((long long)C * H * W);
        long long offset_in_sample = i % ((long long)C * H * W);
        long long c = offset_in_sample / ((long long)H * W);
        // Actually we don't even need n, h, w for PReLU if we just need c.
        // But if we used strides, we would compute offset = n*stride_n + c*stride_c + h*stride_h + w*stride_w
        float val = x[i];
        y[i] = val > 0.0f ? val : weight[c] * val;
    }
}

// Launchers (extern "C" for binding)

extern "C" void custom_softmax_channel_only_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    softmax_channel_only_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W, stride_n, stride_c, stride_h, stride_w);
}

extern "C" void custom_prelu_launcher(
    const float* x,
    float* y,
    const float* weight,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    prelu_kernel<<<blocks, threads, 0, stream>>>(
        x, y, weight, N, C, H, W, stride_n, stride_c, stride_h, stride_w);
}
