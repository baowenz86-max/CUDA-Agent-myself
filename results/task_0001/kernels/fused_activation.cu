#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float mish_nonnegative(float x) {
    // tanh(softplus(x)) = 1 - 2 / ((1 + exp(x))^2 + 1).
    // For x >= 10 the correction is below float precision at this scale.
    if (x >= 10.0f) return x;
    const float e = __expf(x);
    return x * (1.0f - 2.0f / fmaf(e, e + 2.0f, 2.0f));
}

__global__ void fused_square_kernel(float* __restrict__ output,
                                    const float* __restrict__ input,
                                    size_t n,
                                    float negative_slope,
                                    float abs_max) {
    const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        const float4 v = *reinterpret_cast<const float4*>(input + i);
        float4 r;
        float a = v.x < 0.0f ? v.x * negative_slope : v.x;
        float b = v.y < 0.0f ? v.y * negative_slope : v.y;
        float c = v.z < 0.0f ? v.z * negative_slope : v.z;
        float d = v.w < 0.0f ? v.w * negative_slope : v.w;
        r.x = fminf(mish_nonnegative(a * a), abs_max);
        r.y = fminf(mish_nonnegative(b * b), abs_max);
        r.z = fminf(mish_nonnegative(c * c), abs_max);
        r.w = fminf(mish_nonnegative(d * d), abs_max);
        *reinterpret_cast<float4*>(output + i) = r;
    } else {
        for (size_t j = i; j < n; ++j) {
            float a = input[j];
            a = a < 0.0f ? a * negative_slope : a;
            output[j] = fminf(mish_nonnegative(a * a), abs_max);
        }
    }
}

__global__ void fused_general_kernel(float* __restrict__ output,
                                     const float* __restrict__ input,
                                     size_t n,
                                     float negative_slope,
                                     float exponent,
                                     float abs_max) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        float x = input[i];
        x = x < 0.0f ? x * negative_slope : x;
        x = fabsf(powf(x, exponent));
        output[i] = fminf(mish_nonnegative(x), abs_max);
    }
}

}  // namespace

extern "C" void fused_activation_launcher(float* output, const float* input,
                                            size_t n, float negative_slope,
                                            float exponent, float abs_max,
                                            cudaStream_t stream) {
    constexpr int threads = 256;
    if (exponent == 2.0f) {
        const size_t vectors = (n + 3) / 4;
        const int blocks = static_cast<int>((vectors + threads - 1) / threads);
        fused_square_kernel<<<blocks, threads, 0, stream>>>(
            output, input, n, negative_slope, abs_max);
    } else {
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        fused_general_kernel<<<blocks, threads, 0, stream>>>(
            output, input, n, negative_slope, exponent, abs_max);
    }
}
