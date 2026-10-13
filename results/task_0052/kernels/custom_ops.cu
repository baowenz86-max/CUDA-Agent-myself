// custom_ops.cu: CUDA kernels for custom_round, custom_interpolate, custom_argmax
// Author: optimized for performance
// All kernels are pure CUDA, no torch/ATen dependencies.

#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------
// 1. custom_round: element-wise round to nearest integer (float32)
// ---------------------------------------------------------------------
__global__ void custom_round_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = roundf(input[idx]);
    }
}

extern "C" void custom_round_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    // Cap grid size to avoid overflow; kernel uses grid-stride loop
    if (blocks > 1048576) blocks = 1048576;
    custom_round_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, numel);
}

// ---------------------------------------------------------------------
// 2. custom_interpolate: bilinear interpolation, scale_factor=2.0
//    Input: (N, C, H_in, W_in) contiguous float32
//    Output: (N, C, H_out, W_out) where H_out = 2*H_in, W_out = 2*W_in
//    align_corners=False.
// ---------------------------------------------------------------------
__global__ void custom_interpolate_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int N, int C, int H_in, int W_in,
                                          int H_out, int W_out) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H_out * W_out;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const float scale_h = (float)H_in / H_out;
    const float scale_w = (float)W_in / W_out;

    for (; idx < total; idx += stride) {
        int w_out = idx % W_out;
        int64_t tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // compute source coordinates in input pixel space
        // PyTorch with align_corners=False: src = (dst + 0.5) * scale - 0.5
        float src_h = (h_out + 0.5f) * scale_h - 0.5f;
        float src_w = (w_out + 0.5f) * scale_w - 0.5f;

        // clamp negative to 0 (PyTorch uses max(0, src))
        src_h = fmaxf(src_h, 0.0f);
        src_w = fmaxf(src_w, 0.0f);

        int h0 = (int)floorf(src_h);
        int w0 = (int)floorf(src_w);
        int h1 = min(h0 + 1, H_in - 1);
        int w1 = min(w0 + 1, W_in - 1);

        float h_lambda = src_h - h0;
        float w_lambda = src_w - w0;

        // base pointer for this n,c
        const float* in_ptr = input + ((int64_t)n * C + c) * H_in * W_in;

        float val00 = in_ptr[h0 * W_in + w0];
        float val01 = in_ptr[h0 * W_in + w1];
        float val10 = in_ptr[h1 * W_in + w0];
        float val11 = in_ptr[h1 * W_in + w1];

        float val = (1 - h_lambda) * (1 - w_lambda) * val00 +
                    (1 - h_lambda) * w_lambda * val01 +
                    h_lambda * (1 - w_lambda) * val10 +
                    h_lambda * w_lambda * val11;

        output[idx] = val;
    }
}

extern "C" void custom_interpolate_launcher(const float* input,
                                            float* output,
                                            int N, int C, int H_in, int W_in,
                                            int H_out, int W_out,
                                            cudaStream_t stream) {
    int64_t total = (int64_t)N * C * H_out * W_out;
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 1048576) blocks = 1048576;
    custom_interpolate_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out);
}

// ---------------------------------------------------------------------
// 3. custom_argmax: argmax along dim=0 for 4D tensor (N, C, H, W)
//    Input: (N, C, H, W) contiguous float32
//    Output: (C, H, W) int64, each element is index in [0, N-1]
//    This kernel uses one thread per output element (C*H*W), scanning over N.
//    If N is large, use a reduction per spatial location.
// ---------------------------------------------------------------------
__global__ void custom_argmax_dim0_kernel(const float* __restrict__ input,
                                          int64_t* __restrict__ output,
                                          int N, int C, int H, int W) {
    // Each thread handles one (c, h, w) position
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int64_t tmp = idx / W;
    int h = tmp % H;
    int c = tmp / H;

    const float* base = input + ((int64_t)c * H + h) * W + w;
    float max_val = base[0];
    int max_idx = 0;
    for (int n = 1; n < N; ++n) {
        float val = base[(int64_t)n * C * H * W];
        if (val > max_val) {
            max_val = val;
            max_idx = n;
        }
    }
    output[idx] = (int64_t)max_idx;
}

extern "C" void custom_argmax_launcher(const float* input,
                                        int64_t* output,
                                        int N, int C, int H, int W,
                                        cudaStream_t stream) {
    int64_t total = (int64_t)C * H * W;
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 1048576) blocks = 1048576;
    custom_argmax_dim0_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}
