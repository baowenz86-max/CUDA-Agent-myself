#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

// Apply tanh(attn_output * scaling_factor) element-wise.
// Supports float, double, half, and bfloat16 (by converting to float).
template <typename T, typename ComputeT>
__global__ void tanh_scale_kernel_template(
    T* __restrict__ output,
    const T* __restrict__ input,
    float scaling_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ComputeT val = static_cast<ComputeT>(input[idx]);
        ComputeT out = static_cast<ComputeT>(std::tanh(val * static_cast<ComputeT>(scaling_factor)));
        output[idx] = static_cast<T>(out);
    }
}

// Specialized for half (converting via float)
template <>
__global__ void tanh_scale_kernel_template<__half, float>(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    float scaling_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(input[idx]);
        float out = tanhf(val * scaling_factor);
        output[idx] = __float2half(out);
    }
}

// Specialized for bfloat16 (converting via float)
template <>
__global__ void tanh_scale_kernel_template<__nv_bfloat16, float>(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    float scaling_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(input[idx]);
        float out = tanhf(val * scaling_factor);
        output[idx] = __float2bfloat16(out);
    }
}

// Explicit specialization for float
template <>
__global__ void tanh_scale_kernel_template<float, float>(
    float* __restrict__ output,
    const float* __restrict__ input,
    float scaling_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        float out = tanhf(val * scaling_factor);
        output[idx] = out;
    }
}

// Explicit specialization for double
template <>
__global__ void tanh_scale_kernel_template<double, double>(
    double* __restrict__ output,
    const double* __restrict__ input,
    float scaling_factor,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = input[idx];
        double out = tanh(val * static_cast<double>(scaling_factor));
        output[idx] = out;
    }
}

// Host launcher for tanh scaling. dtype_size: 2 (half/bfloat16), 4 (float), 8 (double)
extern "C" void tanh_scale_launcher(
    void* output,
    const void* input,
    float scaling_factor,
    int n,
    int dtype_size,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    switch (dtype_size) {
        case 2:
            // NOTE: The dtype_size alone cannot distinguish half from bfloat16.
            // This launcher assumes that 2-byte types are half only.
            // To support bfloat16, we would need additional dtype information,
            // but for this task we only expect float32.
            tanh_scale_kernel_template<__half, float><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<__half*>(output),
                reinterpret_cast<const __half*>(input),
                scaling_factor, n);
            break;
        case 4:
            tanh_scale_kernel_template<float, float><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<float*>(output),
                reinterpret_cast<const float*>(input),
                scaling_factor, n);
            break;
        case 8:
            tanh_scale_kernel_template<double, double><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<double*>(output),
                reinterpret_cast<const double*>(input),
                scaling_factor, n);
            break;
        default:
            break;
    }
}
