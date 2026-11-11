// custom_ops.cu
// CUDA kernels and launchers for custom_abs_forward and custom_flip_forward.
// No PyTorch dependencies. All tensors are passed as raw pointers with metadata.

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Element-wise absolute value kernel
// ---------------------------------------------------------------------------
__global__ void abs_kernel(float* __restrict__ out, const float* __restrict__ in, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = fabsf(in[i]);
    }
}

// ---------------------------------------------------------------------------
// Flip kernel: flips along the two specified dimensions (dims and dim2).
// The input tensor is assumed to be contiguous and float32. The shape and strides
// are passed as arrays (host-side) but for contiguous tensor we can compute
// offsets directly based on shape. However, to support potential non-contiguous
// inputs (contract says "supports strides"), we pass both shape and strides.
// For simplicity and correctness, we assume the input is contiguous as per
// the actual reference data. The kernel uses stride-based indexing.
// ---------------------------------------------------------------------------
__global__ void flip_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t dims,        // first flip dimension (e.g., 2)
    int64_t dim2,        // second flip dimension (e.g., 3)
    const int64_t* __restrict__ shape,   // array of ndim elements (host pointer, but copied to device?)
    const int64_t* __restrict__ strides, // array of ndim elements
    int64_t ndim,
    int64_t numel
) {
    // Since ndim=4 and shape/strides are small, we can copy them into registers
    // or use constant memory. For simplicity, each thread reads them directly.
    // We assume the arrays are in device memory (passed via cudaMemcpyAsync from binding).
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    int64_t s[8];  // support up to 8 dims
    for (int64_t d = 0; d < ndim; ++d) {
        s[d] = shape[d];
    }

    for (int64_t i = idx; i < numel; i += stride) {
        // Compute multi-dimensional indices from linear index i (row-major, assuming contiguous output)
        int64_t idx_arr[8];
        int64_t rem = i;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            idx_arr[d] = rem % s[d];
            rem /= s[d];
        }

        // Apply flip on dims and dim2: new_idx = (s[d] - 1 - idx_arr[d]) for flipped dims
        idx_arr[dims] = s[dims] - 1 - idx_arr[dims];
        idx_arr[dim2] = s[dim2] - 1 - idx_arr[dim2];

        // Compute input offset using strides (assuming input is contiguous, but strides may be standard)
        int64_t in_offset = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            in_offset += idx_arr[d] * strides[d];
        }

        out[i] = in[in_offset];
    }
}

// ---------------------------------------------------------------------------
// Launcher: custom_abs_launcher
// Called from binding.cpp after obtaining raw pointers and stream.
// ---------------------------------------------------------------------------
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid to avoid excessive launch config (max grid dim is 2^31-1)
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    abs_kernel<<<(int)grid_size, block_size, 0, stream>>>(output, input, numel);
}

// ---------------------------------------------------------------------------
// Launcher: custom_flip_launcher
// Parameters:
//   output: device pointer to output tensor (contiguous)
//   input: device pointer to input tensor (contiguous)
//   dims: first flip dimension
//   dim2: second flip dimension
//   shape: device pointer to array of ndim int64 shape values (copied by binding)
//   strides: device pointer to array of ndim int64 stride values (copied by binding)
//   ndim: number of dimensions
//   numel: total number of elements
//   stream: CUDA stream
// Note: binding must allocate device memory for shape and strides and copy them.
// ---------------------------------------------------------------------------
extern "C" void custom_flip_launcher(
    float* output,
    const float* input,
    int64_t dims,
    int64_t dim2,
    const int64_t* shape,
    const int64_t* strides,
    int64_t ndim,
    int64_t numel,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    flip_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        output, input, dims, dim2, shape, strides, ndim, numel
    );
}
