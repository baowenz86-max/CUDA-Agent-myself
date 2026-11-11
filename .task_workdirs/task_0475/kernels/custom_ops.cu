#include <cuda_runtime.h>
#include <cmath>

// Simple identity copy kernel: output[i] = input[i] for contiguous tensors.
__global__ void identity_copy_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total_elements
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        output[idx] = input[idx];
    }
}

// ELU kernel: y = x if x > 0 else alpha * (exp(x) - 1)
__global__ void elu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float alpha,
    long long total_elements
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

extern "C" void custom_identity_launcher(
    float* output,
    const float* input,
    long long total_elements,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576;
    identity_copy_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements);
}

extern "C" void custom_elu_launcher(
    float* output,
    const float* input,
    float alpha,
    long long total_elements,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576;
    elu_kernel<<<blocks, threads, 0, stream>>>(output, input, alpha, total_elements);
}
