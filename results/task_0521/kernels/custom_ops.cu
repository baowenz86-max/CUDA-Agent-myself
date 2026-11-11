// kernels/custom_ops.cu - CUDA kernels for custom PyTorch operations
// Contains embedding forward, randn_like, and smooth_l1_loss kernels with C-style launchers.
// No torch/ATen/c10/pybind dependencies. Binding layer will call these launchers.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Embedding forward kernel
// Gather rows from weight according to indices. Supports non-contiguous weight
// via row stride in elements.
// weight: float pointer, shape [num_embeddings, embedding_dim], row stride = weight_row_stride
// indices: int64 pointer, arbitrary shape flattened, total count = total_indices
// output: float pointer, shape [total_indices, embedding_dim], row stride = out_row_stride
// embedding_dim: number of features per row
//
// Kernel configuration: one thread per output element, i.e., total_indices * embedding_dim threads.
__global__ void embedding_forward_kernel(
    const float* __restrict__ weight,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int embedding_dim,
    int64_t weight_row_stride,
    int64_t out_row_stride,
    int64_t total_indices)
{
    // Grid-stride loop for robust large data handling
    int64_t total_elements = total_indices * embedding_dim;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        int64_t row = idx / embedding_dim;
        int col = idx - row * embedding_dim;
        int64_t weight_row = indices[row]; // assumes valid index [0, num_embeddings)
        output[row * out_row_stride + col] = weight[weight_row * weight_row_stride + col];
    }
}

// -----------------------------------------------------------------------------
// Randn_like kernel: generate standard normal random numbers using cuRAND.
// output: float pointer, total_numel elements
// seed: seed for cuRAND, passed to each thread with a unique offset based on global index
__global__ void randn_like_kernel(
    float* __restrict__ output,
    uint64_t seed,
    int64_t total_numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_numel) {
        curandStatePhilox4_32_10_t state;
        // Use unique subsequence per thread by offsetting seed with idx.
        curand_init(seed, idx, 0, &state);
        // Generate 4 normal floats per call for efficiency, but only use one.
        float4 rand4 = curand_normal4(&state);
        output[idx] = rand4.x;
    }
}

// -----------------------------------------------------------------------------
// Smooth L1 loss reduction: computes mean over all elements of loss(input, target)
// loss = 0.5 * (diff)^2 if |diff| < 1 else |diff| - 0.5
// We perform a two-phase reduction: within-block reduce to partial sums, then
// atomicAdd to a single float global accumulator, and finally divide by N in a
// small kernel or by host launcher (host will write scalar output).
// To avoid launch of a separate division kernel, we have the block that finishes
// last perform division and write final result to output scalar.

// Kernel 1: compute per-element loss and block reduction, accumulate to global atomically.
// output_sum: pointer to a float accumulator initialized to 0 on device
// count_valid: pointer to int64 accumulator; not needed if total_numel known, but we still
// keep count for safety? Actually we know N exactly from total_numel, so no need for count.
// We'll just add partial sums to output_sum atomically.
__global__ void smooth_l1_loss_partial_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ partial_sum,
    int64_t total_numel,
    float beta_inv) // 1.0 / beta, contract beta=1.0
{
    extern __shared__ float sdata[]; // shared memory for block reduction
    int64_t tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    float my_loss = 0.0f;

    // Grid-stride loop over elements to handle large data and reduce per-thread work
    for (int64_t i = idx; i < total_numel; i += blockDim.x * gridDim.x) {
        float diff = input[i] - target[i];
        float abs_diff = fabsf(diff);
        // smooth l1 with beta=1.0: if abs_diff < 1 -> 0.5*abs_diff^2, else abs_diff - 0.5
        float loss;
        if (abs_diff < 1.0f) {
            loss = 0.5f * abs_diff * abs_diff;
        } else {
            loss = abs_diff - 0.5f;
        }
        my_loss += loss;
    }

    // Block reduction
    sdata[tid] = my_loss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block atomically adds to global partial sum
    if (tid == 0) {
        atomicAdd(partial_sum, sdata[0]);
    }
}

// Kernel 2: finalize mean by dividing partial_sum by total_numel and write scalar output.
__global__ void smooth_l1_loss_finalize_kernel(
    const float* __restrict__ partial_sum,
    float* __restrict__ output,
    int64_t total_numel)
{
    if (threadIdx.x == 0) {
        output[0] = (*partial_sum) / (float)total_numel;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C" so binding.cpp can call without name mangling)
// These assume tensors are already on CUDA and pointers are valid.
// They should be called with the current CUDA stream.

// Embedding forward launcher
// weight_ptr: pointer to weight tensor (float)
// indices_ptr: pointer to indices tensor (int64)
// output_ptr: pointer to allocated output tensor (float)
// embedding_dim: int
// num_indices: total number of indices (int64)
// weight_row_stride: stride between weight rows in elements
// out_row_stride: stride between output rows in elements (usually embedding_dim)
// stream: CUDA stream
extern "C" void custom_embedding_forward_launcher(
    const float* weight_ptr,
    const int64_t* indices_ptr,
    float* output_ptr,
    int embedding_dim,
    int64_t num_indices,
    int64_t weight_row_stride,
    int64_t out_row_stride,
    cudaStream_t stream)
{
    // Launch config: enough threads to cover total elements, grid-stride loop used.
    int64_t total_elements = num_indices * embedding_dim;
    const int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    // Cap grid to avoid exceeding max grid size (though grid-stride handles large)
    if (blocks > 65535) blocks = 65535;

    embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
        weight_ptr, indices_ptr, output_ptr,
        embedding_dim, weight_row_stride, out_row_stride, num_indices);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, this would be turned into a C++ exception;
        // here we just rely on cudaGetLastError being checked by caller after launcher.
        // We'll leave as is but avoid calling cudaDeviceSynchronize here.
    }
}

// Randn_like launcher
// output_ptr: pointer to output tensor (float)
// total_numel: number of elements
// seed: random seed (uint64_t)
// stream: CUDA stream
extern "C" void custom_randn_like_launcher(
    float* output_ptr,
    int64_t total_numel,
    uint64_t seed,
    cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (int)((total_numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    randn_like_kernel<<<blocks, threads, 0, stream>>>(
        output_ptr, seed, total_numel);

    cudaError_t err = cudaGetLastError();
    (void)err;
}

// Smooth L1 loss launcher
// input_ptr: pointer to input tensor (float)
// target_ptr: pointer to target tensor (float)
// output_ptr: pointer to scalar output tensor (float, 0-dim)
// total_numel: number of elements
// stream: CUDA stream
extern "C" void custom_smooth_l1_loss_forward_launcher(
    const float* input_ptr,
    const float* target_ptr,
    float* output_ptr,
    int64_t total_numel,
    cudaStream_t stream)
{
    // Allocate temporary device memory for partial sum (zero-initialized)
    float* partial_sum_dev;
    cudaMallocAsync(&partial_sum_dev, sizeof(float), stream);
    cudaMemsetAsync(partial_sum_dev, 0, sizeof(float), stream);

    const int threads = 256;
    int blocks = (int)((total_numel + threads - 1) / threads);
    // Cap blocks to reasonable number; grid-stride loop handles rest
    int max_blocks = 1024;
    if (blocks > max_blocks) blocks = max_blocks;

    size_t shared_mem_size = threads * sizeof(float);

    smooth_l1_loss_partial_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        input_ptr, target_ptr, partial_sum_dev, total_numel, 1.0f);

    // Finalize: one block of one thread
    smooth_l1_loss_finalize_kernel<<<1, 1, 0, stream>>>(
        partial_sum_dev, output_ptr, total_numel);

    // Clean up temporary memory after finalize completes (using stream-ordered semantics)
    cudaFreeAsync(partial_sum_dev, stream);

    cudaError_t err = cudaGetLastError();
    (void)err;
}
