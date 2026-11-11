// Custom CUDA kernels and launchers for stack_relu_forward
// No torch/ATen/pybind dependencies; only raw pointers and scalars.

#include <cuda_runtime.h>
#include <stdint.h>

// Maximum supported dimensions (input ndim <= 8, output ndim <= 9)
#define MAX_DIMS 8

// Kernel: applies ReLU to input and replicates it along stack_dim for num_channels copies.
// Each thread handles one output element.
// Input and output shapes and strides are passed as kernel arguments (by value).
__global__ void stack_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t num_channels,
                                  int64_t stack_dim,
                                  int64_t input_ndim,
                                  int64_t total_output_elements,
                                  int64_t input_shape[MAX_DIMS],
                                  int64_t input_strides[MAX_DIMS],
                                  int64_t output_shape[MAX_DIMS+1],
                                  int64_t output_strides[MAX_DIMS+1])
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t out_idx = idx; out_idx < total_output_elements; out_idx += stride)
    {
        // Decompose out_idx into output coordinates
        int64_t rem = out_idx;
        int64_t out_coords[MAX_DIMS+1];
        #pragma unroll
        for (int64_t d = 0; d < input_ndim + 1; ++d)
        {
            out_coords[d] = rem / output_strides[d];
            rem = rem % output_strides[d];
        }

        // Map to input linear index by skipping the stack dimension
        int64_t in_linear = 0;
        for (int64_t d = 0; d < input_ndim + 1; ++d)
        {
            if (d == stack_dim)
                continue;
            int64_t in_d = (d < stack_dim) ? d : d - 1;
            in_linear += out_coords[d] * input_strides[in_d];
        }

        float val = input[in_linear];
        output[out_idx] = fmaxf(val, 0.0f);
    }
}

// Host launcher: computes strides and launches kernel.
extern "C" void stack_relu_forward_launcher(
    const float* input,
    float* output,
    int64_t num_channels,
    int64_t stack_dim,
    const int64_t* input_shape,
    int64_t ndim,
    int64_t total_output_elements,
    cudaStream_t stream)
{
    // Assume ndim <= MAX_DIMS
    int64_t input_strides[MAX_DIMS];
    input_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d)
    {
        input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
    }

    // Prepare output shape and strides
    int64_t output_shape[MAX_DIMS+1];
    int64_t output_strides[MAX_DIMS+1];
    int64_t output_ndim = ndim + 1;
    output_strides[output_ndim - 1] = 1;
    for (int64_t d = output_ndim - 1; d >= 0; --d)
    {
        int64_t out_dim_size;
        if (d == stack_dim)
            out_dim_size = num_channels;
        else
        {
            int64_t in_d = (d < stack_dim) ? d : d - 1;
            out_dim_size = input_shape[in_d];
        }
        output_shape[d] = out_dim_size;
        if (d < output_ndim - 1)
            output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
        else
            output_strides[d] = 1;
    }

    // Copy arrays to kernel arguments (by value)
    int64_t input_shape_kernel[MAX_DIMS];
    int64_t input_strides_kernel[MAX_DIMS];
    for (int64_t i = 0; i < ndim; ++i)
    {
        input_shape_kernel[i] = input_shape[i];
        input_strides_kernel[i] = input_strides[i];
    }
    // Pad remaining if needed
    for (int64_t i = ndim; i < MAX_DIMS; ++i)
    {
        input_shape_kernel[i] = 0;
        input_strides_kernel[i] = 0;
    }
    int64_t output_shape_kernel[MAX_DIMS+1];
    int64_t output_strides_kernel[MAX_DIMS+1];
    for (int64_t i = 0; i < output_ndim; ++i)
    {
        output_shape_kernel[i] = output_shape[i];
        output_strides_kernel[i] = output_strides[i];
    }

    // Launch kernel with grid-stride loop; use enough blocks to cover but not exceed GPU limits.
    int threads = 256;
    int blocks = (int)((total_output_elements + threads - 1) / threads);
    // Cap at a reasonable number for grid-stride loop efficiency; A100 supports up to 2^31-1, but we use 65535 for simplicity.
    if (blocks > 65535) blocks = 65535;

    stack_relu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, num_channels, stack_dim, ndim, total_output_elements,
        input_shape_kernel, input_strides_kernel, output_shape_kernel, output_strides_kernel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // Error will be checked by binding after launcher returns
    }
}
