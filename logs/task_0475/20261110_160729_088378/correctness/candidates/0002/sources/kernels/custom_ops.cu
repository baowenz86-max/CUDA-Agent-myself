#include <cuda_runtime.h>
#include <cmath>

// Identity copy kernel: writes output in row-major logical order,
// mapping each logical index through input/output strides.
__global__ void identity_copy_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total_elements,
    int ndim,
    const long long* __restrict__ shape,
    const long long* __restrict__ output_strides,
    const long long* __restrict__ input_strides
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        long long linear = idx;
        long long out_offset = 0;
        long long in_offset = 0;

        // Compute multi-index from row-major linear index and map to offsets.
        // This generic loop supports any ndim.
        for (int d = ndim - 1; d >= 0; --d) {
            long long dim_size = shape[d];
            long long coord = linear % dim_size;
            linear /= dim_size;
            out_offset += coord * output_strides[d];
            in_offset  += coord * input_strides[d];
        }

        output[out_offset] = input[in_offset];
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
    int ndim,
    const long long* shape,
    const long long* output_strides,
    const long long* input_strides,
    long long total_elements,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576;
    identity_copy_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements, ndim, shape, output_strides, input_strides);
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
