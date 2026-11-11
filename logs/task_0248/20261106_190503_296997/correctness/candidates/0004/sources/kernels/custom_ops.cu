// kernels/custom_ops.cu
#include <cuda_runtime.h>
#include <cstdint>

// GELU kernel: elementwise, supports arbitrary strides
__global__ void gelu_kernel_5d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t stride4,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int64_t out_stride3,
    int64_t out_stride4,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t dim4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_stride) {
        int64_t tmp = i;
        int64_t i4 = tmp % dim4; tmp /= dim4;
        int64_t i3 = tmp % dim3; tmp /= dim3;
        int64_t i2 = tmp % dim2; tmp /= dim2;
        int64_t i1 = tmp % dim1; tmp /= dim1;
        int64_t i0 = tmp;
        int64_t in_idx = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3 + i4 * stride4;
        int64_t out_idx = i0 * out_stride0 + i1 * out_stride1 + i2 * out_stride2 + i3 * out_stride3 + i4 * out_stride4;
        float x = input[in_idx];
        float y = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
        output[out_idx] = y;
    }
}

// Bias add kernel for 5D [N, C, D, H, W] with broadcasting bias [C]
__global__ void bias_add_kernel_5d(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t numel,
    int64_t C,
    int64_t inner_size,
    int64_t stride0,
    int64_t stride1,
    int64_t stride2,
    int64_t stride3,
    int64_t stride4,
    int64_t out_stride0,
    int64_t out_stride1,
    int64_t out_stride2,
    int64_t out_stride3,
    int64_t out_stride4,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t dim4) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_stride) {
        int64_t tmp = i;
        int64_t i4 = tmp % dim4; tmp /= dim4;
        int64_t i3 = tmp % dim3; tmp /= dim3;
        int64_t i2 = tmp % dim2; tmp /= dim2;
        int64_t i1 = tmp % dim1; tmp /= dim1;
        int64_t i0 = tmp;
        int64_t in_idx = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3 + i4 * stride4;
        int64_t out_idx = i0 * out_stride0 + i1 * out_stride1 + i2 * out_stride2 + i3 * out_stride3 + i4 * out_stride4;
        output[out_idx] = input[in_idx] + bias[i1];
    }
}

// Launcher for GELU
extern "C" void gelu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
    cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    gelu_kernel_5d<<<blocks, threads, 0, stream>>>(
        input, output, numel,
        in_stride0, in_stride1, in_stride2, in_stride3, in_stride4,
        out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
        dim0, dim1, dim2, dim3, dim4);
}

// Launcher for bias_add
extern "C" void bias_add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t numel,
    int64_t C,
    int64_t D, int64_t H, int64_t W, int64_t N,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
    cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    bias_add_kernel_5d<<<blocks, threads, 0, stream>>>(
        input, bias, output, numel, C, D*H*W,
        in_stride0, in_stride1, in_stride2, in_stride3, in_stride4,
        out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
        N, C, D, H, W);
}
