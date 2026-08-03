#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void fused_activation_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long n,
    float negative_slope,
    float exponent,
    float upper_bound) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)blockDim.x * gridDim.x;

    // The benchmark exponent is exactly two.  Retain a general fallback so
    // the constructor arguments still determine the operation.
    if (exponent == 2.0f) {
        for (long long i = tid; i < n; i += stride) {
            float v = input[i];
            v = v >= 0.0f ? v : v * negative_slope;
            v *= v;
            // For v >= 0, with t=exp(-v):
            // tanh(softplus(v)) = (1 + 2t) / (1 + 2t + 2t^2).
            // This removes log and tanh while remaining algebraically exact.
            const float t = expf(-v);
            const float numerator = fmaf(2.0f, t, 1.0f);
            const float denominator = fmaf(2.0f * t, t, numerator);
            v *= numerator / denominator;
            output[i] = v > upper_bound ? upper_bound : v;
        }
    } else {
        for (long long i = tid; i < n; i += stride) {
            float v = input[i];
            v = v >= 0.0f ? v : v * negative_slope;
            v = fabsf(powf(v, exponent));
            const float softplus = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
            v = v * tanhf(softplus);
            output[i] = fminf(fmaxf(v, 0.0f), upper_bound);
        }
    }
}

extern "C" void fused_activation_launcher(
    float* output,
    const float* input,
    long long n,
    float negative_slope,
    float exponent,
    float upper_bound,
    cudaStream_t stream) {
    constexpr int threads = 256;
    // A persistent grid avoids excessive launch geometry while supplying
    // ample independent work for this large, transcendental-heavy tensor.
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    fused_activation_kernel<<<blocks, threads, 0, stream>>>(
        output, input, n, negative_slope, exponent, upper_bound);
}
