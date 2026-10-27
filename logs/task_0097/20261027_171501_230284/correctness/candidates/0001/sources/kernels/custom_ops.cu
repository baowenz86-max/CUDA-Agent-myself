// custom_ops.cu
// This file contains all custom CUDA kernels and launchers.
//
// Operator implemented in this phase:
//   custom_histc: computes histogram of input elements over [min, max] with given bins.
//
// Operator custom_conv_transpose1d is NOT implemented in this file.
// It must be implemented in a separate .cpp file using cuDNN API,
// per the restriction: Conv/ConvTranspose must use cuDNN and handled by C++ layer.
//
// The launcher custom_histc_launcher is declared extern "C" for linking with binding.

#include <cuda_runtime.h>
#include <math.h>

// Kernel: computes histogram using per-block shared memory private histograms,
// then atomically accumulates into global output. Handles arbitrary number of elements
// using grid-stride loop.
__global__ void custom_histc_kernel(
    const float* __restrict__ input,
    size_t numel,
    int bins,
    float min_val,
    float max_val,
    float* __restrict__ output)
{
    extern __shared__ unsigned int block_hist[]; // shared histogram, one per thread in block
    // initialize shared hist to 0
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        block_hist[i] = 0u;
    }
    __syncthreads();

    float bin_width = (max_val - min_val) / bins;
    if (bin_width <= 0.0f) {
        // invalid range, no elements counted
        __syncthreads(); // ensure all threads before exit? Actually we just return without writing
        return;
    }

    // Loop over all elements assigned to this thread (grid-stride)
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += (size_t)gridDim.x * blockDim.x) {
        float val = input[idx];
        // Ignore NaN and Inf
        if (isnan(val) || isinf(val)) continue;
        // Ignore values outside [min_val, max_val]
        if (val < min_val || val > max_val) continue;

        // Compute bin index
        int bin;
        if (val == max_val) {
            bin = bins - 1; // include max in last bin
        } else {
            bin = (int)((val - min_val) / bin_width);
            // Due to floating precision, bin could equal bins for values very close to max but not equal.
            // Clamp to bins-1 to be safe.
            if (bin >= bins) bin = bins - 1;
            if (bin < 0) bin = 0;
        }
        atomicAdd(&block_hist[bin], 1u);
    }

    __syncthreads();

    // Merge block histogram into global histogram using atomicAdd
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        if (block_hist[i] != 0) {
            atomicAdd(&output[i], (float)block_hist[i]); // output is float, convert
        }
    }
}

// Launcher: custom_histc_launcher
// Parameters:
//   input: pointer to input data (float)
//   numel: number of elements
//   bins: number of bins
//   min_val, max_val: histogram range
//   output: pointer to output array of size bins (float)
//   stream: CUDA stream
// Returns: void, but may set cudaError on failure (kernel launch error can be checked by caller)
extern "C" void custom_histc_launcher(
    const float* input,
    size_t numel,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream)
{
    if (bins <= 0 || numel == 0) {
        // call cudaMemsetAsync to zero the output if necessary? Better to let caller handle; but we can just return.
        return;
    }

    // Determine block size
    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    // Cap grid size to avoid too many blocks; grid-stride loop handles the rest
    const int max_grid_size = 65535;
    if (grid_size > max_grid_size) grid_size = max_grid_size;
    if (grid_size < 1) grid_size = 1;

    // Shared memory per block: bins unsigned ints
    size_t shared_mem = bins * sizeof(unsigned int);
    // Check shared memory limit (typically 48KB, but we assume bins modest)
    // If bins too large, kernel fails; we could fallback, but contract says bins=100.

    custom_histc_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        input, numel, bins, min_val, max_val, output);

    // The caller is expected to check cudaGetLastError() after launch (binding layer).
    // We don't call cudaDeviceSynchronize here; asynchronous error will be caught later.
}
