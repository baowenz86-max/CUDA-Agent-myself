// Custom CUDA kernels and launchers for stack_relu_forward
// No torch/ATen/pybind dependencies; only raw pointers and scalars.

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_DIMS 8

// Kernel: applies ReLU to input and replicates it along stack_dim for num_channels copies.
// Each thread handles one output element.
__global__ void stack_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t input_shape[MAX_DIMS],
    const int64_t input_strides[MAX_DIMS],
    const int64_t output_shape[MAX_DIMS],
    const int64_t output_strides[MAX_DIMS],
    int64_t ndim,
    int64_t stack_dim,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t out_idx = idx; out_idx < total_elements; out_idx += stride)
    {
        // Decompose output linear index into coordinates using output_shape
        int64_t rem = out_idx;
        int64_t in_linear = 0;
        for (int64_t d = 0; d <= ndim; ++d) // output has ndim+1 dims
        {
            int64_t coord = rem / output_strides[d];
            rem = rem % output_strides[d];
            if (d == stack_dim) {
                // stack dimension is ignored for input mapping
                continue;
            }
            int64_t in_d = (d < stack_dim) ? d : (d - 1);
            in_linear += coord * input_strides[in_d];
        }
        float val = input[in_linear];
        output[out_idx] = fmaxf(val, 0.0f);
    }
}

// Host launcher: computes shape/stride arrays on host and passes them by value.
extern "C" void stack_relu_forward_launcher(
    const float* input,
    float* output,
    int64_t num_channels,
    int64_t stack_dim,
    const int64_t* input_shape_host,
    int64_t ndim,
    int64_t total_output_elements,
    cudaStream_t stream)
{
    // Copy input_shape into fixed-size array
    int64_t input_shape[MAX_DIMS] = {0};
    for (int64_t d = 0; d < ndim; ++d) {
        input_shape[d] = input_shape_host[d];
    }

    // Compute input strides (contiguous)
    int64_t input_strides[MAX_DIMS] = {0};
    input_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
        input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
    }

    // Compute output shape (insert num_channels at stack_dim)
    int64_t output_shape[MAX_DIMS] = {0};
    for (int64_t d = 0; d < ndim; ++d) {
        output_shape[d] = input_shape[d];
    }
    // Shift elements after stack_dim to make room
    for (int64_t d = ndim; d > stack_dim; --d) {
        output_shape[d] = input_shape[d - 1];
    }
    output_shape[stack_dim] = num_channels;

    // Compute output strides (contiguous)
    int64_t output_strides[MAX_DIMS] = {0};
    output_strides[ndim] = 1; // ndim+1 dims, last index = ndim
    for (int64_t d = ndim - 1; d >= 0; --d) {
        output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
    }

    // Launch kernel with arrays passed by value (they are copied into kernel param space)
    int threads = 256;
    int blocks = (int)((total_output_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    stack_relu_kernel<<<blocks, threads, 0, stream>>>(
        input,
        output,
        input_shape,
        input_strides,
        output_shape,
        output_strides,
        ndim,
        stack_dim,
        total_output_elements
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked by binding after launcher returns
    }
}
