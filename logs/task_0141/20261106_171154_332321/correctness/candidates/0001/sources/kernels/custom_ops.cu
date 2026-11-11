#include <cuda_runtime.h>

// ============================================================================
// Leaky ReLU kernel
// ============================================================================
// Computes y = x >= 0 ? x : negative_slope * x for each element.
// Input can be non-contiguous; kernel uses strides to compute linear index.
// Output tensor is assumed to have same strides as input (or contiguous, but
// we write using output strides as well). For simplicity, we require both
// input and output to have the same strides, which is guaranteed by the
// binding layer if it creates output with same shape and strides as input.
// However, to be safe we pass both input and output pointers and their
// respective strides. Launcher assumes output was allocated with same strides
// as input.
// Kernel signature: template<int BLOCK_SIZE> __global__ void leaky_relu_kernel(
//   float* out, const float* in, int numel, int ndim,
//   const long long* in_strides, const long long* out_strides, float negative_slope)
// We'll use grid-stride loop for large sizes.

template<int BLOCK_SIZE>
__global__ void leaky_relu_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    long long numel,
    int ndim,
    const long long* __restrict__ in_strides,
    const long long* __restrict__ out_strides,
    float negative_slope)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        // Convert linear index to multi-dimensional index using in_strides
        long long rem = idx;
        long long in_offset = 0;
        long long out_offset = 0;
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            long long coord = rem % out_strides[d] ? rem % out_strides[d] : 0; // not correct; better compute coordinate from shape
        }
        // Instead of computing coordinate from strides directly, we need shape.
        // Since we only have strides, we cannot recover shape for arbitrary dims.
        // Therefore, we require the binding layer to pass the shape as well.
        // For now, we will assume that the input and output are contiguous or
        // have the same shape and we can compute coordinate using the output shape
        // that will be passed separately. Since contract says "layout: supports strides",
        // we need to handle non-contiguous. So we must include shape parameter.
        // We'll adjust launcher to pass shape array.
        // For this implementation, we'll keep a simpler version that assumes
        // contiguous tensors, as the binding layer can make tensors contiguous if
        // needed. But contract explicitly says supports strides, so we should
        // support non-contiguous. We'll add shape array.
        // We'll skip the fully generic non-contiguous implementation for now,
        // but provide a working contiguous path plus a note to binding layer
        // to ensure contiguous? The instructions say "Tensor 检查、分配、当前 stream 获取与 pybind 注册统一放在 kernels/*_binding.cpp",
        // and that .cu should only contain kernels and launchers. It doesn't
        // forbid the binding layer from making tensors contiguous, but contract
        // says layout supports strides, implying we should support non-contiguous
        // without copying. We'll attempt full stride support.
        // For stride support, we need shape. We'll pass shape along.
        out[idx] = (in[idx] >= 0.0f) ? in[idx] : negative_slope * in[idx];
    }
}

// Actually, the above comment and code are messy. Let's rewrite properly.
// We'll implement two versions: one for contiguous (fast), one for general strides.
// But to keep it simple and correct, we'll use a single kernel that takes shape
// and strides, computes offset via manual loop.

__global__ void leaky_relu_generic_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    long long numel,
    int ndim,
    const long long* __restrict__ shape,
    const long long* __restrict__ in_strides,
    const long long* __restrict__ out_strides,
    float negative_slope)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        long long rem = idx;
        long long in_offset = 0;
        long long out_offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            long long coord = rem % shape[d];
            rem /= shape[d];
            in_offset += coord * in_strides[d];
            out_offset += coord * out_strides[d];
        }
        float val = in[in_offset];
        out[out_offset] = (val >= 0.0f) ? val : negative_slope * val;
    }
}

// ============================================================================
// MaxPool2d kernel (kernel_size=2, stride=2, no padding, dilation=1)
// ============================================================================
// Contract: input shape [N, C, H, W], output shape [N, C, H/2, W/2]
// with kernel_size=2, stride=2. We'll support arbitrary H, W, but kernel_size
// and stride fixed to 2? Since parameters are optional and default to 2, we
// can assume kernel_size=2, stride=2 for this task. We'll hardcode for
// performance, but launcher will check and only call for 2x2/2.
// Input can be non-contiguous. We'll use a generic kernel with strides.
// We'll compute output coordinate (n,c,oh,ow) and then input start (n,c,oh*2,ow*2)
// and take max of 4 elements.

__global__ void max_pool2d_2x2_stride2_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    int outH, int outW,
    const long long* __restrict__ in_strides, // length 4: [N,C,H,W]
    const long long* __restrict__ out_strides)
{
    // Use linear index over output elements: index = n*C*outH*outW + c*outH*outW + oh*outW + ow
    long long total_out = (long long)N * C * outH * outW;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_out; idx += stride) {
        int ow = idx % outW;
        long long tmp = idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        int ih = oh * 2;
        int iw = ow * 2;

        // Compute base offset for (n,c,ih,iw)
        long long in_base = (long long)n * in_strides[0] + (long long)c * in_strides[1]
                         + (long long)ih * in_strides[2] + (long long)iw * in_strides[3];

        float v0 = in[in_base];
        float v1 = in[in_base + in_strides[3]]; // iw+1
        float v2 = in[in_base + in_strides[2]]; // ih+1, iw
        float v3 = in[in_base + in_strides[2] + in_strides[3]]; // ih+1, iw+1

        float max_val = v0;
        if (v1 > max_val) max_val = v1;
        if (v2 > max_val) max_val = v2;
        if (v3 > max_val) max_val = v3;

        long long out_offset = (long long)n * out_strides[0] + (long long)c * out_strides[1]
                            + (long long)oh * out_strides[2] + (long long)ow * out_strides[3];
        out[out_offset] = max_val;
    }
}

// ============================================================================
// Launcher functions (extern "C" for binding layer)
// ============================================================================

extern "C" void leaky_relu_launcher(
    float* out,
    const float* in,
    long long numel,
    int ndim,
    const long long* shape,
    const long long* in_strides,
    const long long* out_strides,
    float negative_slope,
    cudaStream_t stream)
{
    // Configure kernel launch
    const int block_size = 256;
    long long blocks = (numel + block_size - 1) / block_size;
    if (blocks > 0) {
        leaky_relu_generic_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
            out, in, numel, ndim, shape, in_strides, out_strides, negative_slope);
    }
}

extern "C" void max_pool2d_2x2_stride2_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int outH, int outW,
    const long long* in_strides,
    const long long* out_strides,
    cudaStream_t stream)
{
    long long total_out = (long long)N * C * outH * outW;
    const int block_size = 256;
    long long blocks = (total_out + block_size - 1) / block_size;
    if (blocks > 0) {
        max_pool2d_2x2_stride2_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
            out, in, N, C, H, W, outH, outW, in_strides, out_strides);
    }
}

// ============================================================================
// Conv2d placeholder: actual cuDNN implementation will be added in phase 3.
// We declare a launcher that will be defined elsewhere, but since this file
// must compile, we provide a stub that does nothing (not used).
// However, the binding layer may call conv2d_launcher, so we need to provide a
// definition. The instruction says Conv must use cuDNN by third stage, but
// we can stub it here. We'll define a dummy that just returns, assuming
// binding layer will not be compiled until phase 3 includes the real one.
// But to be safe, we provide an empty function.

extern "C" void conv2d_launcher(
    float* /*out*/,
    const float* /*input*/,
    const float* /*weight*/,
    const float* /*bias*/,
    int /*N*/, int /*C*/, int /*H*/, int /*W*/,
    int /*K*/, int /*R*/, int /*S*/,
    int /*stride_h*/, int /*stride_w*/,
    int /*pad_h*/, int /*pad_w*/,
    int /*dilation_h*/, int /*dilation_w*/,
    int /*use_bias*/,
    cudaStream_t /*stream*/)
{
    // Placeholder: to be replaced with cuDNN call in phase 3.
    // This function should not be called until then.
}
