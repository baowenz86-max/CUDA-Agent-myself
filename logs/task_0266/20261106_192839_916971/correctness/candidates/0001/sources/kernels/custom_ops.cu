#include <cuda_runtime.h>

// Kernel: sums embedding rows for each bag
// input_indices: int64_t array of all indices (length total_indices)
// offsets: int64_t array of size num_bags+1, where offsets[i] is start of bag i
// weight: float array of size num_embeddings * embedding_dim
// output: float array of size num_bags * embedding_dim
template <int BLOCK_DIM>
__global__ void embedding_bag_sum_kernel(
    const int64_t* __restrict__ input_indices,
    const int64_t* __restrict__ offsets,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_bags,
    int embedding_dim) {
    int bag_idx = blockIdx.x;
    if (bag_idx >= num_bags) return;

    int start = static_cast<int>(offsets[bag_idx]);
    int end = static_cast<int>(offsets[bag_idx + 1]);
    int bag_size = end - start;

    // Each block processes one bag, with threads across embedding_dim
    int tid = threadIdx.x;
    if (tid >= embedding_dim) return;

    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        int64_t idx = input_indices[i];
        sum += weight[idx * embedding_dim + tid];
    }

    output[bag_idx * embedding_dim + tid] = sum;
}

// Launcher function called from binding.cpp
// All pointers are CPU pointers; we allocate GPU memory, copy, launch, and return GPU pointer
// Output GPU pointer is returned via out_ptr parameter allocated by caller? No, caller expects us to allocate.
// Actually binding layer will allocate output tensor on GPU and pass its data_ptr, but here we don't have torch.
// Instead, we define a function that takes raw pointers and handles everything internally.
// Binding layer will call this with CPU input pointers and expects a GPU output tensor.
// Since .cu cannot use torch, we return a float* to GPU memory allocated with cudaMalloc.
// Binding layer will wrap it into a tensor with from_blob? That would require torch in binding.cpp which is allowed.
// So launcher signature: float* embedding_bag_sum_forward_launcher(
//     const int64_t* input_cpu, const int64_t* offsets_cpu, const float* weight_cpu,
//     int total_indices, int num_bags, int embedding_dim, int num_embeddings, cudaStream_t stream)
// It allocates output on GPU, copies inputs, performs work, and returns output pointer.
// Caller (binding) is responsible for freeing with cudaFree later? Better: binding allocates output via torch::empty and passes device pointer.
// Typical pattern: binding allocates output tensor, passes data_ptr<float> to launcher, launcher copies inputs and launches kernel.
// We'll follow that: launcher takes output_device_ptr, input_cpu_ptr, offsets_cpu_ptr, weight_cpu_ptr, sizes, stream.
// It allocates temporary GPU buffers for input, offsets, weight, copies, launches, and frees temporary buffers after kernel completes (async).
// This keeps binding simple and avoids memory ownership issues.

extern "C" void embedding_bag_sum_forward_launcher(
    float* output,                // GPU output pointer (already allocated)
    const int64_t* input_cpu,     // CPU input indices
    const int64_t* offsets_cpu,   // CPU offsets
    const float* weight_cpu,      // CPU weight
    int total_indices,            // length of input
    int num_bags,                 // offsets length - 1
    int embedding_dim,            // embedding dimension
    int num_embeddings,           // number of rows in weight
    cudaStream_t stream) {

    // Allocate GPU memory for copies
    int64_t* input_gpu = nullptr;
    int64_t* offsets_gpu = nullptr;
    float* weight_gpu = nullptr;

    size_t input_size = static_cast<size_t>(total_indices) * sizeof(int64_t);
    size_t offsets_size = static_cast<size_t>(num_bags + 1) * sizeof(int64_t);
    size_t weight_size = static_cast<size_t>(num_embeddings) * embedding_dim * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&input_gpu, input_size);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&offsets_gpu, offsets_size);
    if (err != cudaSuccess) { cudaFree(input_gpu); return; }
    err = cudaMalloc(&weight_gpu, weight_size);
    if (err != cudaSuccess) { cudaFree(input_gpu); cudaFree(offsets_gpu); return; }

    // Asynchronous copies on stream
    err = cudaMemcpyAsync(input_gpu, input_cpu, input_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(input_gpu); cudaFree(offsets_gpu); cudaFree(weight_gpu); return; }
    err = cudaMemcpyAsync(offsets_gpu, offsets_cpu, offsets_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(input_gpu); cudaFree(offsets_gpu); cudaFree(weight_gpu); return; }
    err = cudaMemcpyAsync(weight_gpu, weight_cpu, weight_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(input_gpu); cudaFree(offsets_gpu); cudaFree(weight_gpu); return; }

    // Launch kernel: one block per bag, threads = embedding_dim (assuming embedding_dim <= 1024)
    const int block_dim = 256; // can be up to 1024, but use 256 for occupancy
    // If embedding_dim > 1024, need different strategy, but contract says 5, so fine.
    dim3 grid(num_bags);
    dim3 block(block_dim);

    embedding_bag_sum_kernel<256><<<grid, block, 0, stream>>>(
        input_gpu, offsets_gpu, weight_gpu, output,
        num_bags, embedding_dim);

    // Free GPU temporary buffers after kernel completes (use cudaFree, but may free before kernel done? Use cudaFreeAsync?)
    // Since kernel is async on stream, we can't free immediately because kernel may still be running.
    // Use cudaFree after synchronizing? But we don't want to sync inside launcher.
    // Simplest: record event and free after event? For simplicity, we can use cudaFree after kernel launch, but that may cause issues.
    // Better: allocate via cudaMallocAsync with stream, or use cudaFreeAsync (CUDA 11.2+).
    // Since target may be CUDA 11, we can use cudaFreeAsync if available, else fallback to cudaFree with a sync.
    // We'll call cudaFreeAsync when possible.
#if CUDART_VERSION >= 11020
    cudaFreeAsync(input_gpu, stream);
    cudaFreeAsync(offsets_gpu, stream);
    cudaFreeAsync(weight_gpu, stream);
#else
    // Fallback: cudaFree after ensuring kernel finished (sync).
    cudaStreamSynchronize(stream);
    cudaFree(input_gpu);
    cudaFree(offsets_gpu);
    cudaFree(weight_gpu);
#endif
}
