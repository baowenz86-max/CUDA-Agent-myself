#include <cuda_runtime.h>
#include <stdint.h>

// Element-wise absolute value for contiguous float tensor.
__global__ void custom_abs_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = fabsf(in[i]);
    }
}

// Reduce the last dimension (W) by summing elements.
// Input and output are contiguous. Output has shape [B, C, H].
__global__ void custom_reduce_lastdim_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = B * C * H;
    if (idx >= total) return;

    int64_t tmp = idx;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t c = tmp % C;
    tmp /= C;
    int64_t b = tmp;

    int64_t base = (b * C * H + c * H + h) * W;
    float sum = 0.0f;
    for (int64_t w = 0; w < W; ++w) {
        sum += in[base + w];
    }
    out[idx] = sum;
}

extern "C" void custom_abs_launcher(
    float* out,
    const float* in,
    int64_t numel,
    int64_t block_size,
    int64_t grid_size,
    cudaStream_t stream
) {
    custom_abs_kernel<<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(
        out, in, numel);
}

extern "C" void custom_reduce_lastdim_launcher(
    float* out,
    const float* in,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t block_size,
    int64_t grid_size,
    cudaStream_t stream
) {
    custom_reduce_lastdim_kernel<<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(
        out, in, B, C, H, W);
}
