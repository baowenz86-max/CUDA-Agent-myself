#include <cuda_runtime.h>
#include <stdint.h>

// Computes average and max pooling over non-overlapping windows for 4D tensors.
// Input: x with shape [N, C, H, W], strides [sN, sC, sH, sW]
// Outputs: avg and max with shape [N, C, H_out, W_out], contiguous
// kernel: kh x kw, stride = kernel (non-overlapping), padding 0, dilation 1
extern "C" __global__ void pool2d_avgmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ avg,
    float* __restrict__ max,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t sN,
    int64_t sC,
    int64_t sH,
    int64_t sW,
    int64_t kh,
    int64_t kw
) {
    int64_t H_out = H / kh;
    int64_t W_out = W / kw;
    int64_t total = N * C * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        int64_t wo = linear % W_out;
        int64_t tmp = linear / W_out;
        int64_t ho = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        const float* base = x + n * sN + c * sC + ho * kh * sH + wo * kw * sW;
        float sum = 0.0f;
        float mx = -3.402823466e+38f; // -FLT_MAX, but use C float minimum? We'll initialize with first value.

        for (int64_t i = 0; i < kh; ++i) {
            const float* row = base + i * sH;
            for (int64_t j = 0; j < kw; ++j) {
                float val = row[j * sW];
                sum += val;
                if (i == 0 && j == 0) {
                    mx = val;
                } else if (val > mx) {
                    mx = val;
                }
            }
        }

        float avg_val = sum / (float)(kh * kw);
        int64_t out_idx = linear;
        avg[out_idx] = avg_val;
        max[out_idx] = mx;
    }
}

// Host C-interface launcher (no PyTorch dependencies)
extern "C" void pool2d_avgmax_launcher(
    const float* x,
    float* avg,
    float* max,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t sN,
    int64_t sC,
    int64_t sH,
    int64_t sW,
    int64_t kh,
    int64_t kw,
    cudaStream_t stream
) {
    int64_t H_out = H / kh;
    int64_t W_out = W / kw;
    int64_t total = N * C * H_out * W_out;
    if (total <= 0) return;

    const int block_size = 256;
    // grid cap for safety, use grid-stride loop inside kernel
    int64_t grid = (total + block_size - 1) / block_size;
    if (grid > 65535) grid = 65535;

    pool2d_avgmax_kernel<<<(unsigned int)grid, block_size, 0, stream>>>(
        x, avg, max, N, C, H, W, sN, sC, sH, sW, kh, kw);
}
