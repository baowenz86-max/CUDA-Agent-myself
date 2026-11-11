#include <cuda_runtime.h>

// ELU forward kernel with arbitrary strides.
// Each thread processes one or more linear indices via grid-stride loop.
// Computes out = x if x > 0 else exp(x) - 1 (alpha=1.0).
__global__ void elu_forward_kernel(
    float* output,
    const float* input,
    int rank,
    const int64_t* shape,
    const int64_t* input_strides,
    const int64_t* output_strides,
    int64_t numel)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        // Convert linear index to multidimensional indices using input strides
        int64_t rem = idx;
        int64_t in_offset = 0;
        int64_t out_offset = 0;
        for (int d = 0; d < rank; ++d) {
            int64_t dim_size = shape[d];
            int64_t coord = rem % dim_size;
            rem /= dim_size;
            in_offset += coord * input_strides[d];
            out_offset += coord * output_strides[d];
        }
        float val = input[in_offset];
        output[out_offset] = (val > 0.0f) ? val : (expf(val) - 1.0f);
    }
}

extern "C" void elu_forward_launcher(
    float* output,
    const float* input,
    int rank,
    const int64_t* shape,
    const int64_t* input_strides,
    const int64_t* output_strides,
    int64_t numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid excessive blocks (e.g., 2^31-1 limit)
    const int64_t max_grid_size = 32768; // safe for most GPUs
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }
    if (grid_size == 0) return;
    elu_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        output, input, rank, shape, input_strides, output_strides, numel);
}
