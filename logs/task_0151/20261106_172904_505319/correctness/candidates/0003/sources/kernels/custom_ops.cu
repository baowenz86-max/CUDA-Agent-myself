// CUDA kernels and launchers for custom operations.
// Conv/ConvTranspose use cuDNN and are implemented in the C++ binding file.
// This file contains the element-wise pow kernel and a declaration for the conv_transpose1d launcher.

#include <cuda_runtime.h>
#include <cstdint>

// Forward declaration for conv_transpose1d launcher (implemented in C++ binding with cuDNN).
extern "C" void custom_conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

// Element-wise power kernel with arbitrary strides.
// __restrict__ pointers, assumes output and input do not alias.
template <typename T>
__global__ void pow_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T exponent,
    int numel,
    int ndim,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_grid = blockDim.x * gridDim.x;

    for (int linear = idx; linear < numel; linear += stride_grid) {
        // Compute multi-dimensional index from linear index using shape and strides.
        int remaining = linear;
        int64_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
            int dim_size = static_cast<int>(shape[d]);
            int coord = remaining % dim_size;
            remaining /= dim_size;
            offset += static_cast<int64_t>(coord) * strides[d];
        }
        output[offset] = powf(input[offset], exponent);
    }
}

// Launcher for element-wise power. Input and output are separate tensors.
// ndim is the number of dimensions; shape and strides are arrays of length ndim.
// The output tensor is assumed to have the same shape and strides as input.
// If the output is contiguous (as allocated in binding), strides may differ from input,
// but for simplicity we copy the same strides for output. For correctness with non-contiguous
// input, the binding should allocate a contiguous output and pass its strides here.
extern "C" void custom_pow_launcher(
    float* output,
    const float* input,
    const float exponent,
    int numel,
    int ndim,
    const int64_t* shape,
    const int64_t* strides,
    cudaStream_t stream) {

    if (numel <= 0) return;

    const int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    // Cap grid to avoid too many blocks; grid-stride loop handles the rest.
    const int max_blocks = 65535;
    if (grid_size > max_blocks) grid_size = max_blocks;

    pow_kernel<float><<<grid_size, block_size, 0, stream>>>(
        output, input, exponent, numel, ndim, shape, strides);

    // Error check (optional, but recommended)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
}
