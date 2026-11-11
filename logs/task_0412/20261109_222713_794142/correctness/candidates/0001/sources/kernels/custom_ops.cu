#include <cuda_runtime.h>

// Helper macros for error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            // In production launcher, just report via cudaGetLastError.        \
            // We cannot throw here; binding will check after call.             \
            break;                                                               \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// BatchNorm2d inference (evaluation mode)
// y = weight * (x - running_mean) / sqrt(running_var + eps) + bias
// Each thread handles one element.
// -----------------------------------------------------------------------------
__global__ void batch_norm_eval_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int N, int C, int H, int W,
    float eps
) {
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        // Compute channel index: i % (H*W) / (H*W) but we can compute easier
        int n = (int)(i / ((int64_t)C * H * W));
        int rem = (int)(i % ((int64_t)C * H * W));
        int c = rem / (H * W);

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;

        y[i] = gamma * (x[i] - mean) * inv_std + beta;
    }
}

// -----------------------------------------------------------------------------
// AvgPool2d forward (kernel_size=3 assumed, stride=2, padding=0)
// Each thread computes one output element (n, c, oh, ow).
// -----------------------------------------------------------------------------
__global__ void avg_pool2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int outputH, int outputW,
    int kernel_size, int stride, int padding
) {
    int64_t total = (int64_t)N * C * outputH * outputW;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total; i += stride_grid) {
        int n = (int)(i / ((int64_t)C * outputH * outputW));
        int rem = (int)(i % ((int64_t)C * outputH * outputW));
        int c = rem / (outputH * outputW);
        int ohw = rem % (outputH * outputW);
        int oh = ohw / outputW;
        int ow = ohw % outputW;

        int h_start = oh * stride - padding;
        int w_start = ow * stride - padding;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;
        // With padding=0, boundaries are simple: h_start >= 0 and h_end <= H
        // For general padding, handle bounds.
        int h_begin = max(h_start, 0);
        int w_begin = max(w_start, 0);
        int h_last = min(h_end, H);
        int w_last = min(w_end, W);

        float sum = 0.0f;
        int count = 0;
        for (int h = h_begin; h < h_last; ++h) {
            for (int w = w_begin; w < w_last; ++w) {
                sum += x[((int64_t)n * C + c) * H * W + h * W + w];
                ++count;
            }
        }
        if (count > 0) {
            y[i] = sum / count;
        } else {
            y[i] = 0.0f;  // shouldn't happen with valid kernel/padding
        }
    }
}

// -----------------------------------------------------------------------------
// Launchers exported for binding layer
// -----------------------------------------------------------------------------

extern "C" void custom_batch_norm2d_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C * H * W;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (int)min((int64_t)65535, (total + block_size - 1) / block_size);
    // If total exceeds grid_size*block_size, grid-stride loop covers.

    batch_norm_eval_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var,
        N, C, H, W, eps
    );

    // Binding will call cudaGetLastError after launcher returns.
}

extern "C" void custom_avg_pool2d_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int outputH, int outputW,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C * outputH * outputW;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (int)min((int64_t)65535, (total + block_size - 1) / block_size);

    avg_pool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y,
        N, C, H, W,
        outputH, outputW,
        kernel_size, stride, padding
    );
}
