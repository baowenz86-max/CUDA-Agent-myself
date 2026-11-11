#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// clamp_forward: element-wise clamp to [min_val, max_val]
// Input/output: float32 tensor with arbitrary strides
// -----------------------------------------------------------------------------
__global__ void clamp_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float min_val,
    float max_val)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = fminf(fmaxf(val, min_val), max_val);
    }
}

extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    int block_size = 256;
    int64_t blocks64 = (numel + block_size - 1) / block_size;
    int blocks = (int)min(blocks64, (int64_t)INT32_MAX);
    clamp_forward_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, numel, min_val, max_val);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return; // error will be checked in binding
}

// -----------------------------------------------------------------------------
// softmax_dim1_forward: softmax over dimension 1 (channels)
// Input: float32 tensor with arbitrary strides, shape [N, C, H, W]
// Output: float32 tensor same shape (may be same memory if in-place)
// Implementation: two passes over channels for numerical stability
// Each thread block handles one (n, h, w) location, loops over C
// -----------------------------------------------------------------------------
__global__ void softmax_dim1_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_positions = N * H * W;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total_positions; pos += stride) {
        int64_t n = pos / (H * W);
        int64_t rem = pos % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        const float* base = input + n * stride_n + h * stride_h + w * stride_w;
        float* out_base = output + n * stride_n + h * stride_h + w * stride_w;

        // Find max over channels
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; ++c) {
            float val = base[c * stride_c];
            if (val > max_val) max_val = val;
        }

        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            sum += __expf(base[c * stride_c] - max_val);
        }

        // Normalize and write output
        for (int64_t c = 0; c < C; ++c) {
            out_base[c * stride_c] = __expf(base[c * stride_c] - max_val) / sum;
        }
    }
}

extern "C" void softmax_dim1_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int block_size = 256;
    int64_t total_positions = N * H * W;
    int64_t blocks64 = (total_positions + block_size - 1) / block_size;
    int blocks = (int)min(blocks64, (int64_t)INT32_MAX);
    softmax_dim1_forward_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return;
}

// -----------------------------------------------------------------------------
// slice_channels_forward: copy first max_channels channels from input to output
// Input: float32 tensor with arbitrary strides, shape [N, C_in, H, W]
// Output: float32 contiguous tensor shape [N, max_channels, H, W]
// Assumes output stride: stride_n_out = max_channels*H*W, stride_c_out = H*W,
//                        stride_h_out = W, stride_w_out = 1
// -----------------------------------------------------------------------------
__global__ void slice_channels_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t N,
    int64_t max_channels,
    int64_t C_in,
    int64_t H,
    int64_t W,
    int64_t stride_n_in,
    int64_t stride_c_in,
    int64_t stride_h_in,
    int64_t stride_w_in)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * max_channels * H * W;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        // Decompose linear index into n, c, h, w
        int64_t tmp = i;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % max_channels;
        int64_t n = tmp / max_channels;

        const float* src = input + n * stride_n_in + c * stride_c_in
                         + h * stride_h_in + w * stride_w_in;
        float* dst = output + i; // output is contiguous
        *dst = *src;
    }
}

extern "C" void slice_channels_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t max_channels,
    int64_t C_in,
    int64_t H,
    int64_t W,
    int64_t stride_n_in,
    int64_t stride_c_in,
    int64_t stride_h_in,
    int64_t stride_w_in,
    cudaStream_t stream)
{
    if (N <= 0 || max_channels <= 0 || H <= 0 || W <= 0) return;
    int block_size = 256;
    int64_t total = N * max_channels * H * W;
    int64_t blocks64 = (total + block_size - 1) / block_size;
    int blocks = (int)min(blocks64, (int64_t)INT32_MAX);
    slice_channels_forward_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, N, max_channels, C_in, H, W,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return;
}
