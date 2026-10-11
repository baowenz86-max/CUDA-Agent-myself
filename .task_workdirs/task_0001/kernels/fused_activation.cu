#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Fused activation kernel: LeakyReLU -> Power -> Abs -> Mish -> ReLU -> Clamp
// All operations are element-wise, no shape change.
template <typename scalar_t>
__global__ void fused_activation_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t x = input[idx];

        // LeakyReLU: x if x >= 0 else slope * x
        x = x >= 0 ? x : static_cast<scalar_t>(leakyrelu_slope * x);

        // Power: x^pow_exponent
        x = static_cast<scalar_t>(pow(static_cast<double>(x), pow_exponent));

        // Abs
        x = fabs(x);

        // Mish: x * tanh(softplus(alpha * x))
        double xd = static_cast<double>(x);
        double sp = log1p(exp(mish_alpha * xd));
        xd *= tanh(sp);
        x = static_cast<scalar_t>(xd);

        // ReLU with threshold: max(x, threshold)
        x = fmax(x, static_cast<scalar_t>(relu_threshold));

        // Clamp: min=0, max=abs_max
        x = fmin(fmax(x, static_cast<scalar_t>(0)), static_cast<scalar_t>(abs_max));

        output[idx] = x;
    }
}

// Specialized kernel for half: compute in float, convert back to half.
__global__ void fused_activation_kernel_half(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = __half2float(input[idx]);

        // LeakyReLU
        x = x >= 0.0f ? x : static_cast<float>(leakyrelu_slope * x);

        // Power
        x = static_cast<float>(pow(static_cast<double>(x), pow_exponent));

        // Abs
        x = fabsf(x);

        // Mish
        double xd = static_cast<double>(x);
        double sp = log1p(exp(mish_alpha * xd));
        xd *= tanh(sp);
        x = static_cast<float>(xd);

        // ReLU with threshold
        x = fmaxf(x, static_cast<float>(relu_threshold));

        // Clamp
        x = fminf(fmaxf(x, 0.0f), static_cast<float>(abs_max));

        output[idx] = __float2half(x);
    }
}

// Launcher for float
extern "C" void fused_activation_float_launcher(
    float* output,
    const float* input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    fused_activation_kernel<float><<<blocks, threads, 0, stream>>>(
        output, input, numel,
        leakyrelu_slope, pow_exponent, mish_alpha, relu_threshold, abs_max);
}

// Launcher for double
extern "C" void fused_activation_double_launcher(
    double* output,
    const double* input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    fused_activation_kernel<double><<<blocks, threads, 0, stream>>>(
        output, input, numel,
        leakyrelu_slope, pow_exponent, mish_alpha, relu_threshold, abs_max);
}

// Launcher for half
extern "C" void fused_activation_half_launcher(
    __half* output,
    const __half* input,
    int numel,
    double leakyrelu_slope,
    double pow_exponent,
    double mish_alpha,
    double relu_threshold,
    double abs_max,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    fused_activation_kernel_half<<<blocks, threads, 0, stream>>>(
        output, input, numel,
        leakyrelu_slope, pow_exponent, mish_alpha, relu_threshold, abs_max);
}
