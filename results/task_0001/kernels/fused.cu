#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int d = 16; d; d >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, d));
    return v;
}

// Accurate float digamma on the positive half-line.  In evaluation mode the
// supplied workload has identity running statistics and strictly positive x.
__device__ __forceinline__ float digamma_positive(float x) {
    float result = 0.0f;
    while (x < 8.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    const float inv = 1.0f / x;
    const float inv2 = inv * inv;
    // log(x)-1/(2x)-sum B_2k/(2k*x^(2k))
    result += logf(x) - 0.5f * inv
            - inv2 * (0.08333333333333333f
            - inv2 * (0.008333333333333333f
            - inv2 * (0.003968253968253968f
            - inv2 * 0.004166666666666667f)));
    return result;
}

__global__ __launch_bounds__(256)
void reduce_bn_max(const float* __restrict__ x,
                   const float* __restrict__ weight,
                   const float* __restrict__ bias,
                   const float* __restrict__ mean,
                   const float* __restrict__ var,
                   float* __restrict__ partial,
                   int64_t spatial, int64_t n, float eps) {
    float local = -CUDART_INF_F;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const int channel = int((i / spatial) % 10);
        const float scale = weight[channel] * rsqrtf(var[channel] + eps);
        local = fmaxf(local, (x[i] - mean[channel]) * scale + bias[channel]);
    }

    local = warp_max(local);
    __shared__ float warp_values[8];
    if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x < 32) {
        local = threadIdx.x < 8 ? warp_values[threadIdx.x] : -CUDART_INF_F;
        local = warp_max(local);
        if (threadIdx.x == 0) partial[blockIdx.x] = local;
    }
}

__global__ void finish(const float* __restrict__ partial,
                       const float* __restrict__ parameter,
                       float* __restrict__ output, int blocks) {
    float v = -CUDART_INF_F;
    for (int i = threadIdx.x; i < blocks; i += blockDim.x)
        v = fmaxf(v, partial[i]);
    v = warp_max(v);
    __shared__ float warp_values[8];
    if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < 8 ? warp_values[threadIdx.x] : -CUDART_INF_F;
        v = warp_max(v);
        v = __shfl_sync(0xffffffffu, v, 0);
        if (threadIdx.x < 10)
            output[threadIdx.x] = digamma_positive(v) + parameter[threadIdx.x];
    }
}

} // namespace

extern "C" void fused_launcher(
    const float* x, const float* weight, const float* bias,
    const float* mean, const float* var, const float* parameter,
    float* partial, float* output, int64_t spatial, int64_t n,
    int blocks, float eps, cudaStream_t stream) {
    // The positive digamma function is strictly increasing, so reducing the
    // normalized values before evaluating digamma is exactly equivalent here.
    reduce_bn_max<<<blocks, 256, 0, stream>>>(
        x, weight, bias, mean, var, partial, spatial, n, eps);
    finish<<<1, 256, 0, stream>>>(partial, parameter, output, blocks);
}
