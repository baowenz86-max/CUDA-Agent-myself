#include <cuda_runtime.h>

// CUDA kernel: elementwise embedding lookup followed by greater-than threshold,
// producing a boolean tensor. Output shape = indices.shape + [embedding_dim].
// Thread block size is fixed as 256 for simplicity and good occupancy.
__global__ void embedding_gt_kernel(
    const float* __restrict__ weight,
    const int64_t* __restrict__ indices,
    bool* __restrict__ output,
    int64_t embedding_dim,
    int64_t total_indices,
    double threshold)
{
    int64_t total_output = total_indices * embedding_dim;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
         out_idx < total_output;
         out_idx += stride) {
        int64_t idx = out_idx / embedding_dim;
        int64_t dim = out_idx - idx * embedding_dim;
        int64_t weight_row = indices[idx];
        float val = weight[weight_row * embedding_dim + dim];
        output[out_idx] = (val > (float)threshold);
    }
}

// C-launcher for the embedding_gt_forward custom op.
// All pointers are raw CUDA device pointers; tensors are checked and allocated
// in the corresponding *_binding.cpp file.
extern "C" void embedding_gt_forward_launcher(
    const float* weight,
    const int64_t* indices,
    bool* output,
    int64_t embedding_dim,
    int64_t total_indices,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t total_output = total_indices * embedding_dim;
    // Use a grid large enough to cover all elements, but cap the grid size to
    // avoid excessive launch overhead. The kernel uses a grid-stride loop,
    // so a grid of up to 65535 blocks is sufficient for all realistic sizes.
    int64_t blocks_needed = (total_output + block_size - 1) / block_size;
    int grid_size = (int)(blocks_needed < 65535 ? blocks_needed : 65535);
    if (grid_size == 0) grid_size = 1;

    embedding_gt_kernel<<<grid_size, block_size, 0, stream>>>(
        weight, indices, output, embedding_dim, total_indices, (double)0.5);

    // Launcher does not synchronize; caller may use stream synchronization.
    // Error checking is performed by the binding wrapper after launch.
}
