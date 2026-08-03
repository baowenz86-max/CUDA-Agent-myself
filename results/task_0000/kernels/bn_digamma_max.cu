#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

// Accurate float digamma, following the standard reflection/recurrence/
// asymptotic construction.  Keeping this inline lets the compiler overlap
// independent elements handled by a warp.
__device__ __forceinline__ float digamma_f(float x) {
    if (isnan(x)) return x;
    if (x == 0.0f) return copysignf(CUDART_INF_F, -x);

    float result = 0.0f;
    if (x < 0.0f) {
        // psi(x) = psi(1-x) - pi*cot(pi*x).  Using the fractional part
        // before sin/cos retains precision for arguments far from zero.
        float r = x - floorf(x);
        if (r == 0.0f) return CUDART_NAN_F;
        float a = CUDART_PI_F * r;
        // Reflection: psi(x) = psi(1-x) - pi*cot(pi*x).
        result -= CUDART_PI_F * cosf(a) / sinf(a);
        x = 1.0f - x;
    }
    while (x < 10.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }
    float inv = 1.0f / x;
    float z = inv * inv;
    // log(x)-1/(2x)-sum B_2k/(2k*x^(2k))
    float poly = -1.0f / 12.0f + z * (1.0f / 120.0f + z *
                 (-1.0f / 252.0f + z * (1.0f / 240.0f + z *
                 (-1.0f / 132.0f))));
    return result + logf(x) - 0.5f * inv + z * poly;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, v, d);
        // torch.max propagates NaNs; fmaxf alone deliberately discards them.
        v = (isnan(v) || isnan(other)) ? CUDART_NAN_F : fmaxf(v, other);
    }
    return v;
}

__device__ __forceinline__ void atomic_max_float(float* addr, float value) {
    int* p = reinterpret_cast<int*>(addr);
    int old = *p;
    while (true) {
        float old_value = __int_as_float(old);
        if (isnan(old_value) || (!isnan(value) && old_value >= value)) break;
        int assumed = old;
        old = atomicCAS(p, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void init_max(float* out) {
    if (threadIdx.x == 0) out[0] = -CUDART_INF_F;
}

__global__ __launch_bounds__(256) void bn_digamma_reduce(
    const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ beta, const float* __restrict__ mean,
    const float* __restrict__ var, float* __restrict__ out,
    int64_t n, int64_t spatial, float eps) {
    float local = -CUDART_INF_F;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        int c = (int)((i / spatial) % 10);
        float scale = gamma[c] * rsqrtf(var[c] + eps);
        float y = fmaf(x[i] - mean[c], scale, beta[c]);
        float value = digamma_f(y);
        local = (isnan(local) || isnan(value)) ? CUDART_NAN_F
                                               : fmaxf(local, value);
    }
    local = warp_max(local);
    __shared__ float warp_values[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? warp_values[lane] : -CUDART_INF_F;
        v = warp_max(v);
        if (lane == 0) atomic_max_float(out, v);
    }
}

__global__ void add_parameter(float* out, const float* parameter) {
    __shared__ float maximum;
    if (threadIdx.x == 0) maximum = out[0];
    __syncthreads();
    if (threadIdx.x < 10) out[threadIdx.x] = maximum + parameter[threadIdx.x];
}

extern "C" void bn_digamma_max_launcher(
    const float* x, const float* gamma, const float* beta,
    const float* mean, const float* var, const float* parameter,
    float* output, int64_t n, int64_t spatial, float eps,
    cudaStream_t stream) {
    init_max<<<1, 32, 0, stream>>>(output);
    int blocks = (int)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    bn_digamma_reduce<<<blocks, 256, 0, stream>>>(
        x, gamma, beta, mean, var, output, n, spatial, eps);
    add_parameter<<<1, 32, 0, stream>>>(output, parameter);
}
