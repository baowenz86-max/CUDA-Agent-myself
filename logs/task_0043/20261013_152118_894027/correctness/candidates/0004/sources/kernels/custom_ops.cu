#include <cuda_runtime.h>

// Helper to convert linear index to 4D coordinates (for input with 4 dims)
__device__ __forceinline__ void index_to_4d(
    long long idx,
    long long d1, long long d2, long long d3, long long d4,
    long long &i1, long long &i2, long long &i3, long long &i4)
{
    i4 = idx % d4;
    idx /= d4;
    i3 = idx % d3;
    idx /= d3;
    i2 = idx % d2;
    i1 = idx / d2;
}

// Kernel: view_as_real for complex64 input
// input: complex64 tensor (interpreted as float2 pairs)
// output: float32 tensor with extra last dimension of size 2
// input_shape: [d0,d1,d2] (complex elements)
// input_strides: [s0,s1,s2] in complex elements
// output_strides: [os0,os1,os2,os3] in float elements
__global__ void view_as_real_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long d0, long long d1, long long d2,
    long long s0, long long s1, long long s2,
    long long os0, long long os1, long long os2, long long os3,
    long long total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // idx is linear over output elements: d0*d1*d2*2
        long long out_linear = idx;
        // decompose output linear index into (b,c,h,real_idx)
        long long real_idx = out_linear & 1LL;  // 0 or 1
        long long hw = out_linear >> 1;         // index into d1*d2 elements
        long long c = hw % d2;
        long long h = (hw / d2) % d1;
        long long b = hw / (d2 * d1);
        // input complex element offset (in float units, because input is float pointer to complex data)
        long long in_offset = b * s0 * 2LL + h * s1 * 2LL + c * s2 * 2LL;
        // output offset in float units
        long long out_offset = b * os0 + h * os1 + c * os2 + real_idx * os3;
        output[out_offset] = input[in_offset + real_idx];
    }
}

// Kernel: einsum bchw->bhwc, i.e., transpose last two dims
// input: float32 tensor with shape [B,C,H,W] and strides [is0,is1,is2,is3]
// output: float32 tensor with shape [B,H,W,C] and strides [os0,os1,os2,os3]
__global__ void einsum_bchw_to_bhwc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long B, long long C, long long H, long long W,
    long long is0, long long is1, long long is2, long long is3,
    long long os0, long long os1, long long os2, long long os3,
    long long total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // output linear index -> (b,h,w,c)
        long long c = idx % C;
        long long w = (idx / C) % W;
        long long h = (idx / (C * W)) % H;
        long long b = idx / (C * W * H);
        long long in_offset = b * is0 + c * is1 + h * is2 + w * is3;
        long long out_offset = b * os0 + h * os1 + w * os2 + c * os3;
        output[out_offset] = input[in_offset];
    }
}

// Kernel: clip values to [min_val, max_val]
// input: float32 tensor with arbitrary shape (numel = total) and strides
// output: float32 tensor with same shape, may be arbitrary strides
// We use a 1D linear index over input elements; for each element, compute its
// multidimensional coordinates from the provided shape and strides.
// However, to support arbitrary dim count, we pass shape and strides as arrays.
// For simplicity and to avoid shared memory overhead, we pass the number of dims
// and use dynamic parallelism? No, we can't easily pass variable-length arrays to
// kernel without shared memory. Since this task has 4 dims only, we implement
// specialized 4D version and a generic version? We'll do 4D only, as contract
// specifies 4D.
__global__ void clip_kernel_4d(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long d0, long long d1, long long d2, long long d3,
    long long is0, long long is1, long long is2, long long is3,
    long long os0, long long os1, long long os2, long long os3,
    float min_val, float max_val,
    long long total)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        long long i3 = idx % d3;
        long long rem = idx / d3;
        long long i2 = rem % d2;
        rem /= d2;
        long long i1 = rem % d1;
        long long i0 = rem / d1;
        long long in_offset = i0 * is0 + i1 * is1 + i2 * is2 + i3 * is3;
        long long out_offset = i0 * os0 + i1 * os1 + i2 * os2 + i3 * os3;
        float val = input[in_offset];
        val = fminf(fmaxf(val, min_val), max_val);
        output[out_offset] = val;
    }
}

// Generic clip kernel for any number of dimensions (up to 8) using arrays passed via constant memory or kernel params.
// But to keep it simple, we'll use the 4D version since the contract guarantees 4D.

// Host launchers (extern "C" so binding.cpp can call them)

extern "C" void view_as_real_launcher(
    const float* input,
    float* output,
    long long d0, long long d1, long long d2,
    long long s0, long long s1, long long s2,
    long long os0, long long os1, long long os2, long long os3,
    cudaStream_t stream)
{
    long long total = d0 * d1 * d2 * 2LL;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles the rest
    view_as_real_kernel<<<blocks, threads, 0, stream>>>(
        input, output, d0, d1, d2, s0, s1, s2, os0, os1, os2, os3, total);
}

extern "C" void einsum_bchw_to_bhwc_launcher(
    const float* input,
    float* output,
    long long B, long long C, long long H, long long W,
    long long is0, long long is1, long long is2, long long is3,
    long long os0, long long os1, long long os2, long long os3,
    cudaStream_t stream)
{
    long long total = B * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    einsum_bchw_to_bhwc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, H, W, is0, is1, is2, is3, os0, os1, os2, os3, total);
}

extern "C" void clip_launcher(
    const float* input,
    float* output,
    long long d0, long long d1, long long d2, long long d3,
    long long is0, long long is1, long long is2, long long is3,
    long long os0, long long os1, long long os2, long long os3,
    float min_val, float max_val,
    cudaStream_t stream)
{
    long long total = d0 * d1 * d2 * d3;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    clip_kernel_4d<<<blocks, threads, 0, stream>>>(
        input, output, d0, d1, d2, d3, is0, is1, is2, is3, os0, os1, os2, os3,
        min_val, max_val, total);
}
