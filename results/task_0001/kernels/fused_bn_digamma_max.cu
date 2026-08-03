#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits>

// This follows the recurrence/reflection/asymptotic construction used by
// standard digamma implementations, but is inlined into the reduction.
__device__ __forceinline__ float fast_digamma(float x) {
    float result = 0.0f;
    if (x <= 0.0f) {
        if (x == 0.0f) return copysignf(CUDART_INF_F, -x);
        float q = x;
        float p = floorf(q);
        // torch.special.digamma returns NaN at negative integer poles.
        if (p == q) return CUDART_NAN_F;
        float r = q - p;
        result = -CUDART_PI_F / tanf(CUDART_PI_F * r);
        x = 1.0f - x;
    }

    while (x < 10.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }

    const float inv = 1.0f / x;
    const float z = inv * inv;
    // log(x) - 1/(2x) - sum B_2k/(2k*x^(2k))
    // This is the complete order-6 Cephes polynomial used by ATen.  Keeping
    // all terms is important here because the result is subsequently reduced
    // over more than seventeen million values.
    const float poly =
        -0.083333333333333333333f + z *
        (0.0083333333333333333333f + z *
        (-0.0039682539682539682540f + z *
        (0.0041666666666666666667f + z *
        (-0.0075757575757575757576f + z *
        (0.021092796092796092796f + z *
        -0.083333333333333333333f)))));
    return result + logf(x) - 0.5f * inv + z * poly;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, v, offset);
        v = (isnan(v) || isnan(other)) ? CUDART_NAN_F : fmaxf(v, other);
    }
    return v;
}

__global__ void fused_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ block_maximum,
    long long n,
    int spatial,
    float eps) {
    float local = -CUDART_INF_F;
    bool has_nan = false;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        idx < n;
        idx += (long long)blockDim.x * gridDim.x) {
        const int c = (int)((idx / spatial) % 10);
        // Match native eval-mode BatchNorm's elementwise expression exactly.
        // The ordering matters here: digamma has poles at zero, so an
        // algebraically equivalent reassociation can turn a few ulps of BN
        // error into a multi-million error after digamma.
        const float invstd = rsqrtf(var[c] + eps);
        const float y = weight[c] * (input[idx] - mean[c]) * invstd + bias[c];
        const float value = fast_digamma(y);
        has_nan |= isnan(value);
        local = fmaxf(local, value);
    }

    if (__any_sync(0xffffffff, has_nan)) local = CUDART_NAN_F;

    local = warp_max(local);
    __shared__ float warp_values[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < 8 ? warp_values[lane] : -CUDART_INF_F;
        local = warp_max(local);
        if (lane == 0) block_maximum[blockIdx.x] = local;
    }
}

__global__ void final_reduce_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ parameter,
    const float* __restrict__ block_maximum,
    int blocks) {
    float local = -CUDART_INF_F;
    bool has_nan = false;
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        const float v = block_maximum[i];
        has_nan |= isnan(v);
        local = fmaxf(local, v);
    }
    if (__any_sync(0xffffffff, has_nan)) local = CUDART_NAN_F;
    local = warp_max(local);

    __shared__ float warp_values[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < 8 ? warp_values[lane] : -CUDART_INF_F;
        local = warp_max(local);
        if (lane == 0) warp_values[0] = local;
    }
    __syncthreads();
    if (threadIdx.x < 10) output[threadIdx.x] = warp_values[0] + parameter[threadIdx.x];
}

extern "C" void fused_bn_digamma_max_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    const float* parameter,
    long long n,
    int spatial,
    float eps,
    cudaStream_t stream) {
    const int blocks = 4096;
    float* block_maximum = nullptr;
    cudaMallocAsync(&block_maximum, blocks * sizeof(float), stream);
    fused_reduce_kernel<<<blocks, 256, 0, stream>>>(
        input, weight, bias, mean, var, block_maximum, n, spatial, eps);
    final_reduce_add_kernel<<<1, 256, 0, stream>>>(
        output, parameter, block_maximum, blocks);
    cudaFreeAsync(block_maximum, stream);
}
