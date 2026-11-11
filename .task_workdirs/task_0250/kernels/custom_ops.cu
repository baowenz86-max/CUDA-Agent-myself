#include <cuda_runtime.h>

// Elementwise hardsigmoid kernel supporting arbitrary strides.
// Computes: 0 if x <= -3, 1 if x >= 3, else (x + 3) / 6.
__global__ void hardsigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel,
    int ndims,
    const int* __restrict__ input_strides,
    const int* __restrict__ output_strides) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < numel; linear += stride) {
        // Convert linear index to multi-dimensional index using input strides
        int rem = linear;
        int in_offset = 0;
        int out_offset = 0;
        #pragma unroll
        for (int d = 0; d < ndims; ++d) {
            int coord = rem % input_strides[d];  // not correct for arbitrary strides
            // This simplified approach assumes contiguous dimensions, which is not general.
            // Instead, compute coordinate from linear index and strides properly.
        }
    }
}

// Correct strided access: compute output offset and input offset based on linear index.
// Since input and output share the same shape, we can iterate over linear index and use the same coordinate mapping.
// However, strides may differ, but contract says output has same strides as input, so we can use input strides for both.
__global__ void hardsigmoid_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements,
    int ndims,
    const int* __restrict__ shape,
    const int* __restrict__ strides) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = tid; linear < total_elements; linear += stride) {
        // Compute multi-index from linear index and shape/strides
        int idx = linear;
        int in_offset = 0;
        // Since strides are in elements, and we assume row-major order, compute coordinates:
        // coord_d = (linear / prod(shape[d+1:])) % shape[d]
        // But we can also compute incrementally from last dim.
        int rem = linear;
        int offset = 0;
        #pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int coord = rem % shape[d];
            rem /= shape[d];
            offset += coord * strides[d];
        }
        float val = input[offset];
        float out_val;
        if (val <= -3.0f) {
            out_val = 0.0f;
        } else if (val >= 3.0f) {
            out_val = 1.0f;
        } else {
            out_val = (val + 3.0f) / 6.0f;
        }
        output[offset] = out_val;
    }
}

// Launcher: host function with C linkage for binding.
// Parameters:
//   input: pointer to input float32 tensor (CUDA)
//   output: pointer to output float32 tensor (CUDA, allocated by binding)
//   total_elements: number of elements
//   ndims: number of dimensions
//   shape: pointer to int array of shape (on host), copied to device if needed. In this simple kernel we pass raw pointers directly, but shape/strides must be accessible on device. We'll allocate device memory for them in binding or use constant memory? For simplicity, binding will allocate device memory for shape/strides and copy them before calling launcher.
// For this stage, we provide launcher that expects device pointers for shape and strides, or we can assume contiguous tensor. However contract says supports strides, so we need a generic launcher that can handle non-contiguous input by taking device arrays of shape and strides.
extern "C" void hardsigmoid_forward_launcher(
    const float* input,
    float* output,
    int total_elements,
    int ndims,
    const int* shape_device,
    const int* strides_device,
    cudaStream_t stream) {

    // Launch kernel with grid-stride loop to cover all elements
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to some reasonable max, grid-stride loop handles rest
    int max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    if (blocks == 0) return;

    hardsigmoid_kernel_v2<<<blocks, threads, 0, stream>>>(
        input, output, total_elements, ndims, shape_device, strides_device);
}
