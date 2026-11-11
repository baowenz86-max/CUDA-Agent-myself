#include <cuda_runtime.h>
#include <math.h>

// CUDA kernels and launchers for custom_transpose and custom_selu.
// All launchers use raw pointers, shape/stride info, and cudaStream_t.

// -----------------------------------------------------------------------------
// custom_transpose: swaps dimensions 1 and 2 of a 4D tensor.
// Input shape: [N, C, H, W] -> output shape: [N, H, C, W]
// Supports arbitrary strides (non-contiguous).
// -----------------------------------------------------------------------------
__global__ void transpose_kernel(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t in_stride0, int64_t in_stride1, int64_t in_stride2,
                                 int64_t out_stride0, int64_t out_stride1, int64_t out_stride2,
                                 int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total; linear += grid_size) {
        int64_t w = linear % W;
        int64_t tmp = linear / W;
        int64_t c = tmp % C;
        tmp /= C;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        int64_t src_idx = n * in_stride0 + c * in_stride1 + h * in_stride2 + w;
        int64_t dst_idx = n * out_stride0 + h * out_stride1 + c * out_stride2 + w;
        out[dst_idx] = in[src_idx];
    }
}

extern "C" void custom_transpose_launcher(
    float* out,
    const float* in,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2,
    int64_t total,
    cudaStream_t stream) {

    const int block_size = 256;
    int64_t blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles more
    transpose_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        out, in, N, C, H, W, in_stride0, in_stride1, in_stride2,
        out_stride0, out_stride1, out_stride2, total);
}

// -----------------------------------------------------------------------------
// custom_selu: applies SELU activation elementwise.
// SELU(x) = scale * x if x > 0 else scale * alpha * (exp(x) - 1)
// scale = 1.0507009873554804934193349852946
// alpha = 1.6732632423543772848170429916717
// Supports arbitrary strides.
// -----------------------------------------------------------------------------
__global__ void selu_kernel(float* __restrict__ out,
                            const float* __restrict__ in,
                            int64_t total,
                            int64_t in_stride0, int64_t in_stride1, int64_t in_stride2,
                            int64_t out_stride0, int64_t out_stride1, int64_t out_stride2,
                            int64_t N, int64_t C, int64_t H, int64_t W) {
    const float scale = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total; linear += grid_size) {
        int64_t w = linear % W;
        int64_t tmp = linear / W;
        int64_t c = tmp % C;
        tmp /= C;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        int64_t src_idx = n * in_stride0 + c * in_stride1 + h * in_stride2 + w;
        int64_t dst_idx = n * out_stride0 + c * out_stride1 + h * out_stride2 + w;

        float x = in[src_idx];
        float val = (x > 0.0f) ? (scale * x) : (scale * alpha * (expf(x) - 1.0f));
        out[dst_idx] = val;
    }
}

extern "C" void custom_selu_launcher(
    float* out,
    const float* in,
    int64_t total,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream) {

    const int block_size = 256;
    int64_t blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    selu_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        out, in, total, in_stride0, in_stride1, in_stride2,
        out_stride0, out_stride1, out_stride2, N, C, H, W);
}
