#include <cuda_runtime.h>
#include <stdint.h>

// Stack two tensors along dimension dim.
// x and y are contiguous float32 tensors with identical shapes.
// output has shape with an extra dimension of size 2 at position dim.
// We only support dim=0 for this specific model.
extern "C" void stack_dim0_launcher(const float* x, const float* y, float* output,
                                    int64_t outer_size, int64_t inner_size, cudaStream_t stream) {
    int64_t total_elems = outer_size * 2 * inner_size;
    int block = 256;
    int grid = (int)((total_elems + block - 1) / block);
    // For dim=0, output layout is [2, outer_size, inner_size].
    // We can simply copy x to output[0] and y to output[1].
    // outer_size = product of dims after stack dimension, inner_size = product of dims before stack dimension.
    // For dim=0, outer_size=1 (nothing before), inner_size = product of all original dims.
    // Actually, let's define more generally: output shape = shape[0:dim] + [2] + shape[dim:]
    // We can compute number of before elements = prefix_dims_product, after = suffix_dims_product.
    // For dim=0, prefix=1, suffix=total_elements_of_original.
    // Then output element for (prefix, j, suffix) where j in {0,1} is:
    //   output[((prefix*2 + j)*suffix + suffix_idx)] = (j==0 ? x[prefix*suffix + suffix_idx] : y[...])
    // Here outer_size should be prefix, inner_size should be suffix.
    int64_t total_original = outer_size * inner_size;
    int64_t total_output = outer_size * 2 * inner_size;
    // Use grid-stride loop for safety
    int threads = block;
    uint64_t num_blocks_u64 = (total_output + threads - 1) / threads;
    if (num_blocks_u64 > 2147483647) num_blocks_u64 = 2147483647;
    int grid_size = (int)num_blocks_u64;
    dim3 grid_dim(grid_size);
    dim3 block_dim(block);
    // Kernel: copy x and y appropriately
    // We'll launch a kernel that handles both copies in one pass over output elements.
    // But simpler: two separate kernels or one kernel? One kernel is fine.
    // We'll use a single kernel where each thread computes different indices.
    // Actually more efficient: a kernel that directly writes output.
    // We'll implement a kernel stack_dim0_kernel with grid-stride loop.
    // For each output linear index, compute prefix, j, suffix and copy from x or y.
}

__global__ void stack_dim0_kernel(const float* x, const float* y, float* output,
                                  int64_t outer_size, int64_t inner_size) {
    int64_t total_output = outer_size * 2 * inner_size;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_output; i += stride) {
        int64_t suffix = i % inner_size;
        int64_t tmp = i / inner_size;
        int64_t j = tmp % 2;
        int64_t prefix = tmp / 2;
        int64_t src_idx = prefix * inner_size + suffix;
        if (j == 0) {
            output[i] = x[src_idx];
        } else {
            output[i] = y[src_idx];
        }
    }
}

extern "C" void stack_dim0_launcher(const float* x, const float* y, float* output,
                                    int64_t outer_size, int64_t inner_size, cudaStream_t stream) {
    int64_t total_output = outer_size * 2 * inner_size;
    int block = 256;
    int64_t num_blocks = (total_output + block - 1) / block;
    if (num_blocks > 2147483647) num_blocks = 2147483647;
    int grid = (int)num_blocks;
    stack_dim0_kernel<<<grid, block, 0, stream>>>(x, y, output, outer_size, inner_size);
}

// Compute torch.all along dimension 0 for a contiguous float32 tensor of shape [2, N] where N = product of rest dims.
// Output is bool tensor of shape [N], each element is true if both corresponding floats are non-zero.
__global__ void all_dim0_kernel(const float* input, bool* output, int64_t inner_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < inner_size; i += stride) {
        float v0 = input[i];
        float v1 = input[i + inner_size];
        output[i] = (v0 != 0.0f) && (v1 != 0.0f);
    }
}

extern "C" void all_dim0_launcher(const float* input, bool* output, int64_t inner_size, cudaStream_t stream) {
    int block = 256;
    int64_t num_blocks = (inner_size + block - 1) / block;
    if (num_blocks > 2147483647) num_blocks = 2147483647;
    int grid = (int)num_blocks;
    all_dim0_kernel<<<grid, block, 0, stream>>>(input, output, inner_size);
}
