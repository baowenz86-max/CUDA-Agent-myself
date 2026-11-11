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
    // Assume indices are valid [0, num_embeddings-1]; add clamp for safety but
    // for exact reference behavior no clamping.
    float val = weight[index * embedding_dim + emb_idx];
    output[idx] = val;
}

// Launcher: takes CPU weight and indices, copies to GPU, launches kernel.
// output_ptr must already be allocated on GPU with enough space.
extern "C" void embedding_lookup_launcher(
    const float* weight_cpu,
    const int64_t* indices_cpu,
    float* output_gpu,
    int num_embeddings,
    int height,
    int width,
    int embedding_dim,
    cudaStream_t stream)
{
    // Copy weight and indices to GPU using synchronous cudaMemcpy (or async with stream)
    // For simplicity use cudaMemcpyAsync to stream, but ensure stream synchronization later.
    float* weight_gpu = nullptr;
    int64_t* indices_gpu = nullptr;

    cudaMalloc(&weight_gpu, (size_t)num_embeddings * embedding_dim * sizeof(float));
    cudaMalloc(&indices_gpu, (size_t)height * width * sizeof(int64_t));

    cudaMemcpyAsync(weight_gpu, weight_cpu, (size_t)num_embeddings * embedding_dim * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(indices_gpu, indices_cpu, (size_t)height * width * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);

    int total = height * width * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        weight_gpu, indices_gpu, output_gpu, height, width, embedding_dim);

    // Free temporary GPU buffers after kernel completes.
    // Use cudaFreeAsync if available, otherwise schedule a callback or just cudaFree after stream sync.
    // For simplicity, use cudaFree after cudaStreamSynchronize in caller? But we cannot synchronize here.
    // Better: register a callback to free memory, or use cudaFree with stream capture? 
    // Here we use cudaFreeAsync (CUDA 11.2+ with stream ordered allocation).
    // Since this is a small extension, we can leak for simplicity or use cudaFree after stream sync.
    // We'll use cudaFreeAsync to free after kernel completion.
    #if CUDART_VERSION >= 11020
        cudaFreeAsync((void*)weight_gpu, stream);
        cudaFreeAsync((void*)indices_gpu, stream);
    #else
        // Older CUDA: cannot free async; use cudaFree after stream sync is done externally.
        // We'll just leak for now (acceptable for correctness, not performance).
        // Alternatively, use cudaStreamAddCallback to free.
        cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* userData){
            auto* ptrs = (void**)userData;
            cudaFree(ptrs[0]);
            cudaFree(ptrs[1]);
            delete[] ptrs;
        }, new void*[2]{weight_gpu, indices_gpu}, 0);
    #endif
}
