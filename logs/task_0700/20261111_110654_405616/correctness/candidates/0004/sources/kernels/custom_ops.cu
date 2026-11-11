#include <cuda_runtime.h>

// ==============================================
// CUDA kernels for custom ops (no torch/ATen)
// ==============================================

// ------------------- Unfold kernel -------------------
// Implements nn.Unfold for a 4D input (N, C, H, W) with
// kernel_size (kh, kw), dilation (dh, dw), padding (ph, pw), stride (sh, sw).
// Output shape: (N, C_out, L) where C_out = C * kh * kw, L = out_h * out_w.
__global__ void unfold_kernel_float(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kh, int64_t kw,
    int64_t dh, int64_t dw,
    int64_t ph, int64_t pw,
    int64_t sh, int64_t sw,
    int64_t out_h, int64_t out_w,
    int64_t L,
    int64_t C_out,
    int64_t total_output)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_output; linear_idx += stride) {
        // Decompose linear index: (n, c, l)
        int64_t l = linear_idx % L;
        int64_t tmp = linear_idx / L;
        int64_t c = tmp % C_out;   // fixed: use C_out not C
        int64_t n = tmp / C_out;   // fixed

        int64_t kk = c % (kh * kw);
        int64_t in_c = c / (kh * kw);
        int64_t k_h = kk / kw;
        int64_t k_w = kk % kw;

        int64_t out_i = l / out_w;
        int64_t out_j = l % out_w;

        int64_t in_i = out_i * sh - ph + k_h * dh;
        int64_t in_j = out_j * sw - pw + k_w * dw;

        float val = 0.0f;
        if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W) {
            val = input[((n * C + in_c) * H + in_i) * W + in_j];
        }
        output[linear_idx] = val;
    }
}

// ------------------- Cat kernel -------------------
__global__ void cat_duplicate_kernel_float(
    const float* __restrict__ tensor_a,
    const float* __restrict__ tensor_b,
    float* __restrict__ output,
    int64_t total_elements_per_tensor)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements_per_tensor; i += stride) {
        output[i] = tensor_a[i];
        output[total_elements_per_tensor + i] = tensor_b[i];
    }
}

// ------------------- Launchers -------------------
extern "C" void unfold_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kh, int64_t kw,
    int64_t dh, int64_t dw,
    int64_t ph, int64_t pw,
    int64_t sh, int64_t sw,
    cudaStream_t stream)
{
    int64_t out_h = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t out_w = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    int64_t L = out_h * out_w;
    int64_t C_out = C * kh * kw;
    int64_t total_output = N * C_out * L;

    const int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    if (blocks > 2147483647LL) blocks = 2147483647LL;
    if (total_output > 0) {
        unfold_kernel_float<<<(unsigned int)blocks, threads, 0, stream>>>(
            input, output, N, C, H, W,
            kh, kw, dh, dw, ph, pw, sh, sw,
            out_h, out_w, L, C_out, total_output);
    }
}

extern "C" void cat_forward_launcher(
    const float* tensor_a,
    const float* tensor_b,
    float* output,
    int64_t total_elements_per_tensor,
    cudaStream_t stream)
{
    int64_t total_output = 2 * total_elements_per_tensor;
    const int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    if (blocks > 2147483647LL) blocks = 2147483647LL;
    if (total_output > 0) {
        cat_duplicate_kernel_float<<<(unsigned int)blocks, threads, 0, stream>>>(
            tensor_a, tensor_b, output, total_elements_per_tensor);
    }
}