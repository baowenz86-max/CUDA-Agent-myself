#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void copy_bool_kernel(const bool* __restrict__ src, bool* __restrict__ dst, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void masked_scatter_scan_kernel(
    const float* __restrict__ input,
    const bool* __restrict__ mask,
    const float* __restrict__ source,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t source_numel,
    int* __restrict__ d_block_totals,
    int* __restrict__ d_block_flags)
{
    __shared__ bool s_mask[BLOCK_SIZE];
    __shared__ int s_scan[BLOCK_SIZE];
    __shared__ int s_offset;

    int tid = threadIdx.x;
    int64_t global_idx = (int64_t)blockIdx.x * BLOCK_SIZE + tid;

    if (global_idx < total_elements) {
        s_mask[tid] = mask[global_idx];
    } else {
        s_mask[tid] = false;
    }
    __syncthreads();

    int val = s_mask[tid] ? 1 : 0;
    s_scan[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int n = 0;
        if (tid >= offset) {
            n = s_scan[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            s_scan[tid] += n;
        }
        __syncthreads();
    }

    int block_total = s_scan[BLOCK_SIZE - 1];
    __syncthreads();

    if (tid == 0) {
        d_block_totals[blockIdx.x] = block_total;
        __threadfence();
        atomicExch(&d_block_flags[blockIdx.x], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int offset = 0;
        int bid = blockIdx.x - 1;
        while (bid >= 0) {
            while (atomicAdd(&d_block_flags[bid], 0) == 0) {
                // spin until previous block flag is set
            }
            offset += d_block_totals[bid];
            bid--;
        }
        s_offset = offset;
    }
    __syncthreads();

    if (global_idx < total_elements) {
        if (s_mask[tid]) {
            int exclusive = s_scan[tid] - 1;
            int src_idx = s_offset + exclusive;
            output[global_idx] = source[src_idx];
        } else {
            output[global_idx] = input[global_idx];
        }
    }
}

extern "C" void expand_mask_launcher(
    const bool* input,
    bool* output,
    int64_t numel,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    copy_bool_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void masked_scatter_launcher(
    const float* input,
    const bool* mask,
    const float* source,
    float* output,
    int64_t total_elements,
    int64_t source_numel,
    int* d_block_totals,
    int* d_block_flags,
    cudaStream_t stream)
{
    int grid_size = (int)((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    masked_scatter_scan_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input, mask, source, output, total_elements, source_numel,
        d_block_totals, d_block_flags);
}
