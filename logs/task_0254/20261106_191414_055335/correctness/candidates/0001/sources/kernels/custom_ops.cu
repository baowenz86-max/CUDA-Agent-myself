#include <cuda_runtime.h>

// Element-wise hardtanh (clamp) for float32 tensors with arbitrary strides.
// The launcher computes element offset using the tensor's shape and strides
// by flattening the multi-dimensional index into a linear index and then
// deriving the source/destination offset via position-based multiplication.
// A grid-stride loop ensures all elements are processed even if the total
// number of elements exceeds the maximum grid dimension.
__global__ void hardtanh_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel,
                                int64_t num_dims,
                                const int64_t* __restrict__ shape,
                                const int64_t* __restrict__ input_strides,
                                const int64_t* __restrict__ output_strides,
                                float min_val,
                                float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        // Compute multi-dimensional index from linear index i
        int64_t remaining = i;
        int64_t input_offset = 0;
        int64_t output_offset = 0;
        for (int64_t d = 0; d < num_dims; ++d) {
            int64_t dim_size = shape[d];
            int64_t coord = remaining % dim_size;
            remaining /= dim_size;
            input_offset += coord * input_strides[d];
            output_offset += coord * output_strides[d];
        }

        float val = input[input_offset];
        val = fminf(fmaxf(val, min_val), max_val);
        output[output_offset] = val;
    }
}

// C-interface launcher for hardtanh on CUDA float32 tensors.
// Input and output must be non-null CUDA pointers; dimensions and strides are
// provided by the binding layer from the tensor metadata.
extern "C" void hardtanh_forward_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          int64_t num_dims,
                                          const int64_t* shape,
                                          const int64_t* input_strides,
                                          const int64_t* output_strides,
                                          float min_val,
                                          float max_val,
                                          cudaStream_t stream) {
    if (numel == 0 || input == nullptr || output == nullptr) {
        return;
    }

    // Choose block size and grid size with grid-stride loop
    const int threads = 256;
    int64_t blocks_64 = (numel + threads - 1) / threads;
    // Cap grid to avoid exceeding max grid dimension; remainder handled by loop
    const int64_t max_blocks = 1048576; // typical max grid x dimension
    int blocks = (int)(blocks_64 < max_blocks ? blocks_64 : max_blocks);

    hardtanh_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel, num_dims,
        shape, input_strides, output_strides,
        min_val, max_val);

    // Check for launch errors (but do not synchronize)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, you would throw an exception; here we just
        // return, but the binding should check after launch if necessary.
        // Since .cu cannot throw, we leave it to binding to call cudaGetLastError.
        // We still call it to clear the error status.
    }
}
