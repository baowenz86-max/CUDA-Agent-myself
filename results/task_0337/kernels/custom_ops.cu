#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for embedding lookup.
// Each thread computes one output element (row, col, emb_idx).
// weight: [num_embeddings, embedding_dim] stored row-major on GPU
// indices: [height, width] int64 GPU
// output: [height, width, embedding_dim] float32 GPU
__global__ void embedding_lookup_kernel(
    const float* __restrict__ weight,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int height,
    int width,
    int embedding_dim)
{
    int total = height * width * embedding_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int emb_idx = idx % embedding_dim;
    int spatial = idx / embedding_dim;
    int row = spatial / width;
    int col = spatial % width;

    int64_t index = indices[row * width + col];
    // Assume indices are valid [0, num_embeddings-1]; no clamping for exact behavior.
    float val = weight[index * embedding_dim + emb_idx];
    output[idx] = val;
}

// Launcher: can accept weight and indices either on CPU or GPU.
// weight_on_gpu and indices_on_gpu indicate whether the pointers already point to device memory.
// output_ptr must already be allocated on GPU.
extern "C" void embedding_lookup_launcher(
    const float* weight_ptr,
    bool weight_on_gpu,
    const int64_t* indices_ptr,
    bool indices_on_gpu,
    float* output_gpu,
    int num_embeddings,
    int height,
    int width,
    int embedding_dim,
    cudaStream_t stream)
{
    const float* weight_gpu = nullptr;
    const int64_t* indices_gpu = nullptr;

    if (weight_on_gpu) {
        weight_gpu = weight_ptr;
    } else {
        cudaMalloc(&weight_gpu, (size_t)num_embeddings * embedding_dim * sizeof(float));
        cudaMemcpyAsync((void*)weight_gpu, weight_ptr, (size_t)num_embeddings * embedding_dim * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    if (indices_on_gpu) {
        indices_gpu = indices_ptr;
    } else {
        cudaMalloc(&indices_gpu, (size_t)height * width * sizeof(int64_t));
        cudaMemcpyAsync((void*)indices_gpu, indices_ptr, (size_t)height * width * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
    }

    int total = height * width * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        weight_gpu, indices_gpu, output_gpu, height, width, embedding_dim);

    // Free temporary GPU buffers if they were allocated.
    if (!weight_on_gpu) {
        #if CUDART_VERSION >= 11020
            cudaFreeAsync((void*)weight_gpu, stream);
        #else
            cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* userData){
                cudaFree(userData);
            }, (void*)weight_gpu, 0);
        #endif
    }
    if (!indices_on_gpu) {
        #if CUDART_VERSION >= 11020
            cudaFreeAsync((void*)indices_gpu, stream);
        #else
            cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* userData){
                cudaFree(userData);
            }, (void*)indices_gpu, 0);
        #endif
    }
}
