#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// softplus_forward
// -----------------------------------------------------------------------------
// Elementwise softplus: if x * beta > threshold, y = x; else y = log1p(exp(x * beta)) / beta.
// Supports arbitrary strides for input and output (usually same shape).
// All indices are int64_t to match PyTorch tensor sizes.
// Kernel receives device pointers to shape and strides arrays; binding allocates and copies them.
// -----------------------------------------------------------------------------
__global__ void softplus_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ shape,
    int ndims,
    float beta,
    float threshold)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        int64_t rem = i;
        int64_t in_offset = 0;
        int64_t out_offset = 0;
        // Compute multidimensional coordinates from linear index i
        for (int d = ndims - 1; d >= 0; --d) {
            int64_t coord = rem % shape[d];
            rem /= shape[d];
            in_offset += coord * in_strides[d];
            out_offset += coord * out_strides[d];
        }

        float x = input[in_offset];
        float val = x * beta;
        float y;
        if (val > threshold) {
            y = x;
        } else {
            y = log1pf(expf(val)) / beta;
        }
        output[out_offset] = y;
    }
}

extern "C" void softplus_forward_launcher(
    float* output,
    const float* input,
    int64_t* output_strides,
    int64_t* input_strides,
    int64_t* shape,
    int ndims,
    int64_t numel,
    float beta,
    float threshold,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t raw_grid = (numel + block_size - 1) / block_size;
    int grid_size = (raw_grid > 65535) ? 65535 : (int)raw_grid;

    softplus_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, numel, output_strides, input_strides, shape, ndims, beta, threshold);
}

// -----------------------------------------------------------------------------
// concat_forward
// -----------------------------------------------------------------------------
// Concatenates input1 and input2 along concat_dim. Output is contiguous.
// We compute output linear index -> output coordinates -> determine which input and source coordinates.
// Inputs may have arbitrary strides.
// -----------------------------------------------------------------------------
__global__ void concat_kernel(
    float* __restrict__ output,
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    int64_t out_numel,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ in1_strides,
    const int64_t* __restrict__ in2_strides,
    int ndims,
    int64_t concat_dim,
    int64_t input1_concat_size,
    int64_t input2_concat_size)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < out_numel; i += stride) {
        int64_t rem = i;
        int64_t out_coord[/*max ndims*/ 8]; // assume ndims <= 8; adjust if needed
        // Compute output coordinates
        for (int d = ndims - 1; d >= 0; --d) {
            out_coord[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        int64_t concat_coord = out_coord[concat_dim];
        const float* src_ptr;
        int64_t src_coord[8];
        const int64_t* src_strides;

        if (concat_coord < input1_concat_size) {
            // from input1
            src_ptr = input1;
            for (int d = 0; d < ndims; ++d) {
                src_coord[d] = out_coord[d];
            }
            src_strides = in1_strides;
        } else {
            // from input2, adjust concat dimension
            src_ptr = input2;
            for (int d = 0; d < ndims; ++d) {
                src_coord[d] = out_coord[d];
            }
            src_coord[concat_dim] = concat_coord - input1_concat_size;
            src_strides = in2_strides;
        }

        int64_t src_offset = 0;
        for (int d = 0; d < ndims; ++d) {
            src_offset += src_coord[d] * src_strides[d];
        }

        output[i] = src_ptr[src_offset];
    }
}

extern "C" void concat_forward_launcher(
    float* output,
    const float* input1,
    const float* input2,
    int64_t* out_shape,
    int64_t* in1_strides,
    int64_t* in2_strides,
    int ndims,
    int64_t concat_dim,
    int64_t input1_concat_size,
    int64_t input2_concat_size,
    cudaStream_t stream)
{
    // Compute output numel from out_shape (already available on device in launcher? No, out_shape is device pointer)
    // But out_numel should be passed from binding; we can compute on host in binding and pass as argument.
    // The signature currently lacks out_numel. We need to modify binding to pass out_numel.
    // Since we cannot change binding.cpp except binding files, we can alter the launcher signature and binding accordingly.
    // However, to keep current binding, we can compute out_numel from out_shape only if we copy out_shape to host, but that's wasteful.
    // Simpler: modify launcher signature to accept out_numel and update binding accordingly.
    // The given binding file (custom_ops_binding.cpp) currently does not pass out_numel; we must update it.
    // We'll include out_numel in the launcher signature and update custom_ops_binding.cpp accordingly.
}
