// CUDA kernels for bilinear upsample and squeeze identity
#include <cuda_runtime.h>

// Bilinear interpolation for 4D tensor [N, C, H, W] -> [N, C, H_out, W_out]
// align_corners = false
// scale_factor is float, applied to H and W
// Input and output are contiguous float32 tensors
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    float scale_factor) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * C * H_out * W_out;

    for (int i = index; i < total; i += stride) {
        int w_out = i % W_out;
        int tmp = i / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // source coordinates with align_corners=False
        // Correct formula: src = (dst + 0.5) / scale_factor - 0.5
        float src_h = (static_cast<float>(h_out) + 0.5f) / scale_factor - 0.5f;
        float src_w = (static_cast<float>(w_out) + 0.5f) / scale_factor - 0.5f;

        // clamp to valid range
        if (src_h < 0.0f) src_h = 0.0f;
        if (src_w < 0.0f) src_w = 0.0f;
        if (src_h > H - 1.0f) src_h = H - 1.0f;
        if (src_w > W - 1.0f) src_w = W - 1.0f;

        int h0 = static_cast<int>(src_h);
        int w0 = static_cast<int>(src_w);
        int h1 = (h0 < H - 1) ? h0 + 1 : h0;
        int w1 = (w0 < W - 1) ? w0 + 1 : w0;

        float h_lambda = src_h - h0;
        float w_lambda = src_w - w0;

        // base index for input [n, c, :, :]
        int base = (n * C + c) * H * W;
        float v00 = input[base + h0 * W + w0];
        float v01 = input[base + h0 * W + w1];
        float v10 = input[base + h1 * W + w0];
        float v11 = input[base + h1 * W + w1];

        float top = v00 + (v01 - v00) * w_lambda;
        float bottom = v10 + (v11 - v10) * w_lambda;
        float val = top + (bottom - top) * h_lambda;

        output[i] = val;
    }
}

// Launcher for bilinear upsample
// Expects input and output contiguous, float32, CUDA pointers
extern "C" void bilinear_upsample_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    float scale_factor,
    cudaStream_t stream) {
    int total = N * C * H_out * W_out;
    if (total <= 0) return;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // limit grid to avoid excessive blocks, kernel uses grid-stride loop

    bilinear_upsample_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, scale_factor);
}

// Squeeze identity: torch.squeeze on a 4D tensor with no dimension of size 1
// returns the same tensor (or a view with same shape and stride).
// In our use case, the tensor is [16,32,1024,512] and no dimension is 1,
// so we can simply return the input tensor. No CUDA kernel needed.
// This launcher just returns (does nothing), the binding layer will handle returning input.
extern "C" void squeeze_identity_launcher(
    const float* /*input*/,
    float* /*output*/,
    cudaStream_t /*stream*/) {
    // No operation needed; identity
}
