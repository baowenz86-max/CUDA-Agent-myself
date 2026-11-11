// CUDA kernel for element-wise subtraction of a scalar.
// Only float32 dtype is implemented, as required by the contract.
#include <cuda_runtime.h>

// Kernel: out[i] = in[i] - scalar
// grid-stride loop to handle arbitrary number of elements
__global__ void scalar_subtract_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       float scalar,
                                       long long num_elements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < num_elements; i += stride) {
        output[i] = input[i] - scalar;
    }
}

// Launcher function to be called from the binding layer.
// Parameters:
//   - input: pointer to input tensor data (CUDA float*)
//   - output: pointer to output tensor data (CUDA float*)
//   - scalar: the scalar value (float)
//   - num_elements: total number of elements
//   - stream: CUDA stream to launch kernel on
extern "C" void subtract_scalar_launcher(const float* input,
                                         float* output,
                                         float scalar,
                                         long long num_elements,
                                         cudaStream_t stream) {
    // Use 256 threads per block.
    const int threads = 256;
    // Compute number of blocks, but cap to reasonable number to avoid launch failure
    // while still ensuring coverage via grid-stride loop.
    long long blocks_ll = (num_elements + threads - 1) / threads;
    // Limit blocks to 65535 (max gridDim.x for older GPUs), but since we use grid-stride,
    // a smaller grid still covers all elements. Choose min(blocks_ll, 65535).
    int blocks = (int)(blocks_ll < 65535 ? blocks_ll : 65535);
    if (blocks == 0) {
        blocks = 1;  // launch at least one block even for empty tensors? 
        // Actually for empty tensor num_elements=0, we can skip launch.
        // But to be safe, if num_elements==0, return without launching.
        return;
    }
    scalar_subtract_kernel<<<blocks, threads, 0, stream>>>(input, output, scalar, num_elements);
}
