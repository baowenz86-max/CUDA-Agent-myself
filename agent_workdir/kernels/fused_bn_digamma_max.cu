#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float digamma_float(float x) {
    if (x == 0.0f) {
        return copysignf(CUDART_INF_F, -x);
    }
    if (x < 0.0f && x == truncf(x)) {
        return CUDART_NAN_F;
    }
    float result = 0.0f;
    if (x < 0.0f) {
        // Match ATen exactly here: its float overload intentionally performs
        // argument reduction and tan in double precision.  The maximum over
        // ~17M values is dominated by samples next to digamma's negative
        // integer poles, so a float tan(pi*x) is not sufficiently accurate.
        double integer_part;
        const double fractional = modf(static_cast<double>(x), &integer_part);
        result -= static_cast<float>(CUDART_PI / tan(CUDART_PI * fractional));
        x = 1.0f - x;
    }
    while (x < 10.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    // All values arriving here from this model are advanced to exactly 10.
    // Keeping ATen's special case also avoids approximation drift.
    if (x == 10.0f)
        return result + 2.25175258906672110764f;

    const float coefficients[7] = {
        8.33333333333333333333E-2f, -2.10927960927960927961E-2f,
        7.57575757575757575758E-3f, -4.16666666666666666667E-3f,
        3.96825396825396825397E-3f, -8.33333333333333333333E-3f,
        8.33333333333333333333E-2f};
    const float z = 1.0f / (x * x);
    float polynomial = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i)
        polynomial = fmaf(polynomial, z, coefficients[i]);
    return result + logf(x) - 0.5f / x - z * polynomial;
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void bn_digamma_partial_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ partial,
    int64_t numel,
    int64_t channel_stride) {
    float local = -CUDART_INF_F;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
         index < numel; index += stride) {
        const int channel = static_cast<int>((index / channel_stride) % 10);
        const float scale = weight[channel] * rsqrtf(running_var[channel] + 1.0e-5f);
        const float shift = bias[channel] - running_mean[channel] * scale;
        local = fmaxf(local, digamma_float(fmaf(input[index], scale, shift)));
    }

    local = warp_max(local);
    __shared__ float warp_values[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < BLOCK / 32 ? warp_values[lane] : -CUDART_INF_F;
        local = warp_max(local);
        if (lane == 0) partial[blockIdx.x] = local;
    }
}

template<int BLOCK>
__global__ void finish_max_kernel(const float* __restrict__ partial,
                                  const float* __restrict__ parameter,
                                  float* __restrict__ output, int count) {
    float local = -CUDART_INF_F;
    for (int i = threadIdx.x; i < count; i += BLOCK)
        local = fmaxf(local, partial[i]);
    local = warp_max(local);
    __shared__ float warp_values[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < BLOCK / 32 ? warp_values[lane] : -CUDART_INF_F;
        local = warp_max(local);
        if (lane < 10) output[lane] = __shfl_sync(0xffffffff, local, 0) + parameter[lane];
    }
}

extern "C" void fused_bn_digamma_max_launcher(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, const float* parameter,
    float* partial, float* output, int64_t numel, int64_t channel_stride,
    int partial_count, cudaStream_t stream) {
    constexpr int block = 256;
    bn_digamma_partial_kernel<block><<<partial_count, block, 0, stream>>>(
        input, weight, bias, running_mean, running_var, partial, numel,
        channel_stride);
    finish_max_kernel<block><<<1, block, 0, stream>>>(partial, parameter, output,
                                                      partial_count);
}
