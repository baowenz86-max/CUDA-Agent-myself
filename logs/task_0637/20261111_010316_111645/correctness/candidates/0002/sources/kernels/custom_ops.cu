#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

// CUDA kernel for computing max along a specified dimension for a contiguous tensor.
// Each thread block handles a contiguous range of output elements (one per slice).
// The input is assumed contiguous and of shape given by outer_size, reduce_size, inner_size.
// The output (values) has shape outer_size * inner_size (if keepdim=false) or the same with reduced dim size 1.
// The indices output has the same logical shape as values but with int64 type.
//
// We launch a 1D grid of blocks; each block processes chunks of output elements using grid-stride loop.
// Within a block, threads cooperate to reduce one slice using shared memory.

template <int BLOCK_SIZE>
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size)
{
    // Each block is responsible for one or more output indices, depending on grid-stride.
    // Total output elements = outer_size * inner_size.
    int64_t total_outputs = outer_size * inner_size;
    int64_t block_stride = (int64_t)gridDim.x * blockDim.x;

    // For each output element assigned to this thread (grid-stride loop)
    for (int64_t out_idx_linear = blockIdx.x * blockDim.x + threadIdx.x;
         out_idx_linear < total_outputs;
         out_idx_linear += block_stride)
    {
        // Decode outer and inner indices from linear output index
        int64_t outer = out_idx_linear / inner_size;
        int64_t inner = out_idx_linear - outer * inner_size;

        // Start index in input for the first element of this reduction slice
        int64_t base = (outer * reduce_size) * inner_size + inner;

        // Initialize reduction in registers
        float local_max = -std::numeric_limits<float>::infinity();
        int64_t local_max_idx = 0;

        // Thread-local reduction over the reduce dimension
        for (int64_t r = threadIdx.x; r < reduce_size; r += BLOCK_SIZE)
        {
            float val = input[base + r * inner_size];
            if (val > local_max || (val == local_max && r < local_max_idx)) {
                local_max = val;
                local_max_idx = r;
            }
        }

        // Each thread writes its result directly; no shared memory needed since each thread handles one output element.
        values[out_idx_linear] = local_max;
        indices[out_idx_linear] = local_max_idx;
    }
}

// Launcher function: the binding layer will call this with contiguous inputs and allocated outputs.
// dim: the dimension to reduce (0 <= dim < ndim)
// keepdim: whether to keep the reduced dimension.
// We assume input is contiguous, values and indices are allocated with the correct shape.
// The caller provides pointers and the sizes of the input and output.
extern "C" void max_forward_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int64_t* input_shape,
    int64_t ndim,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream)
{
    if (ndim <= 0) return; // no elements

    // Compute outer_size, reduce_size, inner_size based on dim.
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= input_shape[i];
    int64_t reduce_size = input_shape[dim];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= input_shape[i];

    int64_t total_outputs = outer_size * inner_size;
    if (total_outputs == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int64_t blocks_needed = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_size = (int)min(blocks_needed, (int64_t)65535 * 32); // cap to avoid excessive grid size; grid-stride handles the rest
    if (grid_size < 1) grid_size = 1;

    max_reduce_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input, values, indices, outer_size, reduce_size, inner_size);
}
