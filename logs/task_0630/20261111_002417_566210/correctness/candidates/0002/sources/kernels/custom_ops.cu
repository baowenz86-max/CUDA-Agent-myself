#include <cuda_runtime.h>
#include <stdint.h>

// Performs element-wise absolute value on float tensor.
// Input and output can have arbitrary strides, but same shape and dtype.
// grid-stride loop for large tensors.
__global__ void custom_abs_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t numel,
    int64_t in_offset,
    int64_t out_offset,
    int64_t N,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_strides
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        // Compute multi-dimensional index from linear index i.
        // We assume tensors are at most 4D here (B,C,H,W) but general enough.
        int64_t tmp = i;
        int rem0 = tmp % N;
        tmp /= N;
        int rem1 = tmp % N;
        tmp /= N;
        int rem2 = tmp % N;
        tmp /= N;
        int rem3 = tmp;
        // For actual dimension count <=4, we use the first dimensions.
        // But to be fully generic, we would need ndim. Since contract is 4D,
        // we hardcode 4D. If input rank differs, behavior undefined.
        // Compute input and output linear offsets using strides array (4 entries).
        // The strides array is passed as pointer to 4 int64 values.
        int64_t in_off = in_offset + rem3 * in_strides[0] + rem2 * in_strides[1] + rem1 * in_strides[2] + rem0 * in_strides[3];
        int64_t out_off = out_offset + rem3 * out_strides[0] + rem2 * out_strides[1] + rem1 * out_strides[2] + rem0 * out_strides[3];
        out[out_off] = fabsf(in[in_off]);
    }
}

// Reduces last dimension (W) by summing elements.
// out shape is [B, C, H], contiguous. Input may be non-contiguous with strides.
__global__ void custom_reduce_lastdim_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t in_offset,
    const int64_t* __restrict__ in_strides  // size 4: [B_stride, C_stride, H_stride, W_stride]
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = B * C * H;
    if (idx >= total) return;
    // Decode idx to b, c, h
    int64_t tmp = idx;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t c = tmp % C;
    tmp /= C;
    int64_t b = tmp;
    // Compute base input offset for (b,c,h,0)
    int64_t base = in_offset + b * in_strides[0] + c * in_strides[1] + h * in_strides[2];
    float sum = 0.0f;
    for (int64_t w = 0; w < W; ++w) {
        sum += in[base + w * in_strides[3]];
    }
    out[idx] = sum;
}

extern "C" void custom_abs_launcher(
    float* out,
    const float* in,
    int64_t numel,
    int64_t in_offset,
    int64_t out_offset,
    int64_t N,
    const int64_t* in_strides,
    const int64_t* out_strides,
    int64_t block_size,
    int64_t grid_size,
    cudaStream_t stream
) {
    custom_abs_kernel<<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(
        out, in, numel, in_offset, out_offset, N, in_strides, out_strides);
}

extern "C" void custom_reduce_lastdim_launcher(
    float* out,
    const float* in,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t in_offset,
    const int64_t* in_strides,
    int64_t block_size,
    int64_t grid_size,
    cudaStream_t stream
) {
    custom_reduce_lastdim_kernel<<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(
        out, in, B, C, H, W, in_offset, in_strides);
}
