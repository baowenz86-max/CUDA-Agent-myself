#include <cuda_runtime.h>

// Numerically stable log_softmax along a specified dimension.
// Template instantiated for float only (as used by the model).
// The input is assumed contiguous (as guaranteed by the binding layer).
// dim is the dimension along which to compute log_softmax, 0 <= dim < ndims.
// Output has the same shape and strides as input.
template <typename scalar_t>
__global__ void log_softmax_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int outer_size,
    int reduction_size,
    int inner_size,
    long total_positions)
{
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)blockDim.x * gridDim.x;

    for (; idx < total_positions; idx += stride) {
        // Decompose flat index into (outer, inner)
        int inner = idx % inner_size;
        long outer = idx / inner_size;

        const scalar_t* in_base = input + outer * (long)reduction_size * inner_size + inner;
        scalar_t* out_base = output + outer * (long)reduction_size * inner_size + inner;

        // Find max along reduction dimension
        scalar_t max_val = in_base[0];
        for (int r = 1; r < reduction_size; ++r) {
            scalar_t v = in_base[(long)r * inner_size];
            if (v > max_val) max_val = v;
        }

        // Compute sum of exp(x - max)
        scalar_t sum_exp = 0;
        for (int r = 0; r < reduction_size; ++r) {
            scalar_t v = in_base[(long)r * inner_size] - max_val;
            sum_exp += expf(v);
        }
        scalar_t log_sum_exp = logf(sum_exp);

        // Write log_softmax
        for (int r = 0; r < reduction_size; ++r) {
            scalar_t v = in_base[(long)r * inner_size] - max_val - log_sum_exp;
            out_base[(long)r * inner_size] = v;
        }
    }
}

// Launcher for log_softmax. dims is an array of size ndims with the tensor shape.
// All dimensions are positive and fit in int; total elements can be larger.
// `dim` is the reduction dimension (same as Python dim).
extern "C" void log_softmax_forward_launcher(
    float* output,
    const float* input,
    const int64_t* dims,
    int ndims,
    int dim,
    cudaStream_t stream)
{
    // Normalize negative dim
    if (dim < 0) dim += ndims;

    // Compute outer_size, reduction_size, inner_size
    long outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= dims[i];
    long reduction_size = dims[dim];
    long inner_size = 1;
    for (int i = dim + 1; i < ndims; ++i) inner_size *= dims[i];

    long total_positions = outer_size * inner_size;
    if (total_positions == 0) {
        // Empty tensor: nothing to do
        return;
    }

    // Launch configuration: enough blocks to cover positions, but cap at 65535*256 threads
    // using grid-stride loop.
    const int block = 256;
    long blocks_needed = (total_positions + block - 1) / block;
    int grid = (int)min((long)65535, blocks_needed);

    log_softmax_kernel<float><<<grid, block, 0, stream>>>(
        output, input,
        (int)outer_size,
        (int)reduction_size,
        (int)inner_size,
        total_positions);
}
