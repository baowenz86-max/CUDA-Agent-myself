#include <cuda_runtime.h>
#include <cstdint>

// Kernel: sum over dimension 1 (size 3)
// input x: pointer to float32 tensor with shape [N, 3, H, W]
// strides: sx_N, sx_C, sx_H, sx_W (in elements)
// output: pointer to float32 tensor with shape [N, H, W], contiguous
// total_output = N * H * W
__global__ void sum_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t sx_N,
    int64_t sx_C,
    int64_t sx_H,
    int64_t sx_W,
    int64_t total_output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output; i += stride) {
        int64_t n = i / (H * W);
        int64_t rem = i - n * (H * W);
        int64_t h = rem / W;
        int64_t w = rem - h * W;
        float sum = 0.0f;
        for (int64_t c = 0; c < 3; ++c) {
            sum += x[n * sx_N + c * sx_C + h * sx_H + w * sx_W];
        }
        out[i] = sum;
    }
}

extern "C" void sum_dim1_launcher(
    const float* x,
    float* out,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t sx_N,
    int64_t sx_C,
    int64_t sx_H,
    int64_t sx_W,
    int64_t total_output,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, out, N, H, W, sx_N, sx_C, sx_H, sx_W, total_output);
}

// Kernel: log_softmax over dimension 0 (batch dimension)
// input x: pointer to float32 tensor with shape [N, H, W] and contiguous strides
// output: same shape, contiguous
// total_spatial = H * W
// One thread per spatial location (h,w) computes log_softmax across N
__global__ void log_softmax_dim0_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t total_spatial)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t s = idx; s < total_spatial; s += stride) {
        int64_t h = s / W;
        int64_t w = s - h * W;
        // compute max over N
        float max_val = x[h * W + w]; // index 0
        for (int64_t n = 1; n < N; ++n) {
            float val = x[n * H * W + h * W + w];
            if (val > max_val) max_val = val;
        }
        // compute sum exp(x - max)
        float sum_exp = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            sum_exp += expf(x[n * H * W + h * W + w] - max_val);
        }
        float log_sum = logf(sum_exp);
        // write output: x_i - max - log_sum
        for (int64_t n = 0; n < N; ++n) {
            out[n * H * W + h * W + w] = x[n * H * W + h * W + w] - max_val - log_sum;
        }
    }
}

extern "C" void log_softmax_dim0_launcher(
    const float* x,
    float* out,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t total_spatial,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_spatial + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    log_softmax_dim0_kernel<<<blocks, threads, 0, stream>>>(
        x, out, N, H, W, total_spatial);
}
