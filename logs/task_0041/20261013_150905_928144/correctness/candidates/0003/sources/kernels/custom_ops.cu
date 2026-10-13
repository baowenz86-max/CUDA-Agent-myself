#include <cuda_runtime.h>
#include <stdint.h>

// Element-wise negation: y = -x
__global__ void neg_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = -x[i];
    }
}

// Adaptive average pooling 1D: average over windows defined by start=floor(i*T/O), end=ceil((i+1)*T/O)
// Input: x of shape [B, C, T] with strides (stride_b, stride_c, stride_t)
// Output: y of shape [B, C, O], contiguous
template<int BLOCK_SIZE>
__global__ void adaptive_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B, int64_t C, int64_t T, int64_t O,
    int64_t stride_b, int64_t stride_c, int64_t stride_t)
{
    int64_t total_windows = B * C * O;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t window_idx = idx; window_idx < total_windows; window_idx += grid_stride) {
        int64_t o = window_idx % O;
        int64_t tmp = window_idx / O;
        int64_t c = tmp % C;
        int64_t b = tmp / C;
        
        int64_t start = (o * T) / O;
        int64_t end = ((o + 1) * T + O - 1) / O; // ceil division
        if (end > T) end = T;
        if (start >= end) {
            output[window_idx] = 0.0f;
            continue;
        }
        
        float sum = 0.0f;
        const float* base = input + b * stride_b + c * stride_c;
        for (int64_t t = start; t < end; ++t) {
            sum += base[t * stride_t];
        }
        output[window_idx] = sum / (float)(end - start);
    }
}

// addcmul with broadcasting: out[b,c,o] = bias[o] + t1[b,c,o] * t2[b,c,o]
// t1 and t2 have shape [B, C, O] with strides (t1_stride_b, t1_stride_c, t1_stride_o) etc.
// bias has length O, contiguous.
__global__ void addcmul_tt_kernel(
    const float* __restrict__ bias,
    const float* __restrict__ t1,
    const float* __restrict__ t2,
    float* __restrict__ out,
    int64_t B, int64_t C, int64_t O,
    int64_t t1_stride_b, int64_t t1_stride_c, int64_t t1_stride_o,
    int64_t t2_stride_b, int64_t t2_stride_c, int64_t t2_stride_o)
{
    int64_t total = B * C * O;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t o = i % O;
        int64_t tmp = i / O;
        int64_t c = tmp % C;
        int64_t b = tmp / C;
        int64_t t1_idx = b * t1_stride_b + c * t1_stride_c + o * t1_stride_o;
        int64_t t2_idx = b * t2_stride_b + c * t2_stride_c + o * t2_stride_o;
        out[i] = bias[o] + t1[t1_idx] * t2[t2_idx];
    }
}

// Launcher functions (C linkage)

extern "C" void custom_neg_launcher(
    const float* x,
    float* out,
    int64_t numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535; // cap grid; grid-stride loop handles rest
    neg_kernel<<<blocks, block_size, 0, stream>>>(x, out, numel);
}

extern "C" void custom_adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int64_t B, int64_t C, int64_t T, int64_t O,
    int64_t stride_b, int64_t stride_c, int64_t stride_t,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t total_windows = B * C * O;
    int64_t blocks = (total_windows + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    adaptive_avg_pool1d_kernel<block_size><<<blocks, block_size, 0, stream>>>(
        input, output, B, C, T, O, stride_b, stride_c, stride_t);
}

extern "C" void custom_addcmul_tt_launcher(
    const float* bias,
    const float* t1,
    const float* t2,
    float* out,
    int64_t B, int64_t C, int64_t O,
    int64_t t1_stride_b, int64_t t1_stride_c, int64_t t1_stride_o,
    int64_t t2_stride_b, int64_t t2_stride_c, int64_t t2_stride_o,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t total = B * C * O;
    int64_t blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    addcmul_tt_kernel<<<blocks, block_size, 0, stream>>>(
        bias, t1, t2, out, B, C, O,
        t1_stride_b, t1_stride_c, t1_stride_o,
        t2_stride_b, t2_stride_c, t2_stride_o);
}
