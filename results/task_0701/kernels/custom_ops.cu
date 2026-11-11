#include <cuda_runtime.h>

// Sum reduction over arbitrary-strided float32 tensor, output scalar.
// The launcher signature is designed for binding to call.

#define BLOCK_SIZE 256

__global__ void sum_reduce_kernel(const float* __restrict__ input,
                                  const int64_t* __restrict__ dims,
                                  const int64_t* __restrict__ strides,
                                  int64_t ndim,
                                  int64_t numel,
                                  float* __restrict__ partial_sums) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    float local_sum = 0.0f;
    
    // grid-stride loop for large numel, but each block sums a contiguous chunk of linear indices
    while (idx < numel) {
        // compute the multi-dimensional offset from linear index idx
        int64_t offset = 0;
        int64_t remaining = idx;
        for (int d = 0; d < ndim; ++d) {
            int64_t dim_size = dims[d];
            if (dim_size > 1) {
                int64_t coord = remaining % dim_size;
                offset += coord * strides[d];
                remaining /= dim_size;
            } else {
                // skip dimension of size 1
                offset += 0;
            }
        }
        local_sum += input[offset];
        idx += gridDim.x * blockDim.x;
    }

    // block-level reduction
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(partial_sums, sdata[0]);
    }
}

extern "C" void sum_launcher(const float* input,
                             const int64_t* dims,
                             const int64_t* strides,
                             int64_t ndim,
                             int64_t numel,
                             float* output,
                             cudaStream_t stream) {
    if (numel <= 0) {
        // zero elements -> sum = 0
        cudaMemsetAsync(output, 0, sizeof(float), stream);
        return;
    }

    // Allocate temporary partial sums buffer (1 float)
    float* d_partial = nullptr;
    cudaMalloc(&d_partial, sizeof(float));
    cudaMemsetAsync(d_partial, 0, sizeof(float), stream);

    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Cap blocks to avoid excessive launch overhead? but grid-stride handles.
    // We can set blocks to min(blocks, some max like 1024) but for large numel use grid-stride with enough blocks.
    // Actually we use gridDim.x * blockDim.x in stride, so need to set blocks appropriately.
    // For simplicity, use as many blocks as needed up to a reasonable max (e.g., 32k).
    // but if numel is huge, we want enough parallelism. We'll use min(blocks, 1024) to keep grid size reasonable;
    // grid-stride loop will cover all elements.
    int max_blocks = 1024;
    blocks = min(blocks, max_blocks);

    dim3 grid(blocks);
    dim3 block(BLOCK_SIZE);
    int shared_mem_size = BLOCK_SIZE * sizeof(float);

    sum_reduce_kernel<<<grid, block, shared_mem_size, stream>>>(
        input, dims, strides, ndim, numel, d_partial);

    // Copy partial sum to output (single value); output assumed allocated
    cudaMemcpyAsync(output, d_partial, sizeof(float), cudaMemcpyDeviceToDevice, stream);

    cudaFree(d_partial);
}
