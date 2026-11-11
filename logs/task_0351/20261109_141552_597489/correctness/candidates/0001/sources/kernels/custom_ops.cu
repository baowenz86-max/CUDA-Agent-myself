// CUDA kernels for embedding_forward and scalar_mul_forward
// This file contains only CUDA kernels and launchers with raw pointers,
// scalars and cudaStream_t. No torch/ATen/pybind dependencies.

#include <cuda_runtime.h>
#include <stdint.h>

// Embedding lookup kernel:
// input: int64 indices of shape [B, L] (total elements = batch * length)
// weight: float32 embedding table of shape [num_embeddings, embedding_dim]
// output: float32 tensor of shape [B, L, embedding_dim], contiguous
// Each thread processes one output element.
__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_input_elements,
    int64_t embedding_dim) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_input_elements * embedding_dim; idx += stride) {
        // Determine which input index and which embedding component this thread handles
        int64_t input_offset = idx / embedding_dim;
        int64_t dim_offset = idx - input_offset * embedding_dim;

        int64_t token_idx = input[input_offset];
        // No bounds checking: mimics PyTorch behavior, undefined for out-of-range indices.
        output[input_offset * embedding_dim + dim_offset] = weight[token_idx * embedding_dim + dim_offset];
    }
}

// Scalar multiplication kernel:
// input: float32 contiguous tensor of total numel elements
// output: float32 contiguous tensor of same shape
// Each thread processes one element.
__global__ void scalar_mul_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    int64_t numel) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < numel; idx += stride) {
        output[idx] = input[idx] * scalar;
    }
}

// Launcher for embedding_forward
// Parameters:
//   input: pointer to int64 indices (size = B*L)
//   weight: pointer to float32 embedding table (size = num_embeddings * embedding_dim)
//   output: pointer to float32 output (size = B*L*embedding_dim), already allocated
//   batch: number of rows in input (not used directly except for total elements)
//   length: number of columns in input
//   embedding_dim: dimension of each embedding vector
//   stream: CUDA stream to launch kernel on
// Note: total input elements = batch * length. The caller must ensure pointers are valid.
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t batch,
    int64_t length,
    int64_t embedding_dim,
    cudaStream_t stream) {

    int64_t total_input_elements = batch * length;
    int64_t total_output_elements = total_input_elements * embedding_dim;

    if (total_input_elements == 0 || embedding_dim == 0 || total_output_elements == 0) {
        return;
    }

    // Use a 1D grid. Choose threads and blocks to cover all output elements.
    int threads = 256;
    // Ensure grid does not exceed limits (use int for grid dim, cast from int64_t)
    int64_t blocks_64 = (total_output_elements + threads - 1) / threads;
    // Cap blocks to a reasonable value; grid-stride loop handles excess work.
    int blocks = (blocks_64 > 65535) ? 65535 : static_cast<int>(blocks_64);

    embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output, total_input_elements, embedding_dim);
}

// Launcher for scalar_mul_forward
// Parameters:
//   input: pointer to float32 input tensor (numel elements)
//   output: pointer to float32 output tensor (same numel), already allocated
//   scalar: float multiplier
//   numel: number of elements in input/output
//   stream: CUDA stream to launch kernel on
extern "C" void scalar_mul_forward_launcher(
    const float* input,
    float* output,
    float scalar,
    int64_t numel,
    cudaStream_t stream) {

    if (numel == 0) {
        return;
    }

    int threads = 256;
    int64_t blocks_64 = (numel + threads - 1) / threads;
    int blocks = (blocks_64 > 65535) ? 65535 : static_cast<int>(blocks_64);

    scalar_mul_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scalar, numel);
}
