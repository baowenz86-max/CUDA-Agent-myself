// CUDA kernels and launchers for custom_embedding and custom_split
// This file contains no torch/ATen/pybind dependencies.

#include <cuda_runtime.h>
#include <cstdint>

// Kernel: given int64 indices and float32 weight (num_embeddings x embedding_dim),
// output is N x embedding_dim, each row copied from the corresponding embedding row.
// We use a 1D grid over total output elements (N * embedding_dim) for coalescing.
__global__ void embed_forward_kernel(const int64_t* __restrict__ indices,
                                     const float* __restrict__ weight,
                                     float* __restrict__ output,
                                     int64_t N,
                                     int64_t embedding_dim) {
    int64_t total = N * embedding_dim;
    // grid-stride loop
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t row = idx / embedding_dim;
        int64_t col = idx % embedding_dim;
        int64_t index = indices[row];
        output[idx] = weight[index * embedding_dim + col];
    }
}

// Launcher for custom_embedding. The binding layer is responsible for
// allocating output and passing data pointers.
extern "C" void custom_embedding_launcher(const int64_t* indices,
                                          const float* weight,
                                          float* output,
                                          int64_t N,
                                          int64_t embedding_dim,
                                          cudaStream_t stream) {
    if (N <= 0 || embedding_dim <= 0) return;
    int64_t total = N * embedding_dim;
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid excessive blocks, rely on grid-stride loop.
    const int64_t max_grid = 65535;
    if (grid_size > max_grid) grid_size = max_grid;
    embed_forward_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        indices, weight, output, N, embedding_dim);
}

// Launcher for custom_split. This does NOT launch a kernel; it computes
// pointers and sizes for each split segment. The binding layer uses these
// to create tensor views with torch::from_blob. Since the contract requires
// views sharing storage, the input pointer is returned offset by row dim.
// We assume dim=0 and num_splits=2, as specified in the contract.
extern "C" void custom_split_launcher(const float* input,
                                      int64_t num_splits,
                                      int64_t dim,
                                      int64_t N,
                                      int64_t embedding_dim,
                                      const float** out_ptrs,
                                      int64_t* out_sizes,
                                      cudaStream_t stream) {
    // Only handle dim=0 and num_splits=2 per contract. If other values appear,
    // the binding layer should have already validated.
    if (dim != 0 || num_splits != 2) {
        // In a real implementation, the binding would reject earlier. Here we
        // set outputs to null/0 to signal error, but we cannot throw from .cu.
        out_ptrs[0] = nullptr;
        out_ptrs[1] = nullptr;
        out_sizes[0] = 0;
        out_sizes[1] = 0;
        return;
    }
    int64_t split_size = N / num_splits;
    // First split starts at input.
    out_ptrs[0] = input;
    out_sizes[0] = split_size;
    // Second split starts at input + split_size * embedding_dim (row-major, dim=0).
    out_ptrs[1] = input + split_size * embedding_dim;
    out_sizes[1] = split_size;
}
