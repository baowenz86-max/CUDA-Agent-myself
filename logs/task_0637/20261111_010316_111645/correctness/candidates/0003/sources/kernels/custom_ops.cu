#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

// CUDA kernel for computing max along a specified dimension for a contiguous tensor.
// Each block handles one output element (one slice reduction).
// Input is assumed contiguous. Shape: outer_size x reduce_size x inner_size.
// Output elements: outer_size * inner_size.
// We launch one block per output element; within a block, threads reduce the slice using shared memory.

template <int BLOCK_SIZE>
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size)
{
    // One block per output element.
    int64_t out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int64_t outer = out_idx / inner_size;
    int64_t inner = out_idx - outer * inner_size;
    int64_t base = (outer * reduce_size) * inner_size + inner;

    // Shared memory for reduction: values and indices
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];

    // Each thread loads a strided set of elements into registers and initializes shared memory
    float local_max = -std::numeric_limits<float>::infinity();
    int64_t local_idx = -1;

    for (int64_t r = threadIdx.x; r < reduce_size; r += BLOCK_SIZE) {
        float val = input[base + r * inner_size];
        if (val > local_max) {
            local_max = val;
            local_idx = r;
        } else if (val == local_max && r < local_idx) {
            // tie-breaking: smaller index wins
            local_idx = r;
        }
    }

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_idx;
    __syncthreads();

    // Reduce shared memory using tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other_max = s_max[threadIdx.x + stride];
            int64_t other_idx = s_idx[threadIdx.x + stride];
            if (other_max > s_max[threadIdx.x] ||
                (other_max == s_max[threadIdx.x] && other_idx < s_idx[threadIdx.x])) {
                s_max[threadIdx.x] = other_max;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        values[out_idx] = s_max[0];
        indices[out_idx] = s_idx[0];
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
    // Grid size equals total_outputs, capped if too large; but for this task total_outputs is manageable.
    int grid_size = (int)total_outputs;
    if (grid_size > 65535 * 32) grid_size = 65535 * 32;

    max_reduce_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input, values, indices, outer_size, reduce_size, inner_size);
}
