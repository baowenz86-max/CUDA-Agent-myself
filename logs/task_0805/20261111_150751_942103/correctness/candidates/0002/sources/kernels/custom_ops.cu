// Custom CUDA kernels and launchers for stack_relu_forward
// No torch/ATen/pybind dependencies; only raw pointers and scalars.

#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: applies ReLU to input and replicates it along stack_dim for num_channels copies.
// Each thread handles one output element.
__global__ void stack_relu_kernel_final(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int64_t* __restrict__ out_to_in_stride_map,
                                        const int64_t* __restrict__ output_strides,
                                        int64_t ndim,
                                        int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t out_idx = idx; out_idx < total_elements; out_idx += stride)
    {
        int64_t rem = out_idx;
        int64_t in_linear = 0;
        for (int64_t d = 0; d <= ndim; ++d)
        {
            int64_t coord = rem / output_strides[d];
            rem = rem % output_strides[d];
            in_linear += coord * out_to_in_stride_map[d];
        }
        float val = input[in_linear];
        output[out_idx] = fmaxf(val, 0.0f);
    }
}

// Host launcher: computes necessary strides and launches kernel.
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
    // Compute input strides (contiguous)
    int64_t input_strides[8];
    input_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d)
    {
        input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
    }

    // Compute output contiguous strides
    int64_t output_strides_host[8];
    output_strides_host[ndim] = 1;
    for (int64_t d = ndim - 1; d >= 0; --d)
    {
        int64_t out_dim_size;
        if (d == stack_dim)
            out_dim_size = num_channels;
        else
        {
            int64_t in_d = (d < stack_dim) ? d : (d - 1);
            out_dim_size = input_shape[in_d];
        }
        output_strides_host[d] = output_strides_host[d + 1] * out_dim_size;
    }

    // Compute mapping from output coordinate to input linear index contribution
    int64_t out_to_in_stride_map_host[8];
    for (int64_t d = 0; d <= ndim; ++d)
    {
        if (d == stack_dim)
            out_to_in_stride_map_host[d] = 0;
        else
        {
            int64_t in_d = (d < stack_dim) ? d : (d - 1);
            out_to_in_stride_map_host[d] = input_strides[in_d];
        }
    }

    // Allocate device memory for strides and map
    int64_t* d_output_strides;
    int64_t* d_out_to_in_stride_map;
    cudaMalloc(&d_output_strides, sizeof(int64_t) * (ndim + 1));
    cudaMalloc(&d_out_to_in_stride_map, sizeof(int64_t) * (ndim + 1));
    cudaMemcpyAsync(d_output_strides, output_strides_host, sizeof(int64_t) * (ndim + 1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_to_in_stride_map, out_to_in_stride_map_host, sizeof(int64_t) * (ndim + 1), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    int threads = 256;
    int blocks = (int)((total_output_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    stack_relu_kernel_final<<<blocks, threads, 0, stream>>>(
        input, output, d_out_to_in_stride_map, d_output_strides, ndim, total_output_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // Error will be checked by binding after launcher returns
    }

    // Free device memory after kernel completes (blocking, but small overhead)
    cudaFree(d_out_to_in_stride_map);
    cudaFree(d_output_strides);
}
