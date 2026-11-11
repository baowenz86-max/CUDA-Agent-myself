#include <cuda_runtime.h>
#include <math_constants.h>

#define MAX_BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(old_val, value);
        old = atomicCAS(addr_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
}

template <int BLOCK_SIZE>
__global__ void reduce_max_block_kernel(const float* __restrict__ input,
                                       float* __restrict__ block_max,
                                       long long numel) {
    __shared__ float smem[BLOCK_SIZE];

    long long tid = blockIdx.x * (long long)BLOCK_SIZE + threadIdx.x;
    long long stride = (long long)BLOCK_SIZE * gridDim.x;

    float local_max = -INFINITY;

    for (long long i = tid; i < numel; i += stride) {
        float val = input[i];
        local_max = fmaxf(local_max, val);
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}

__global__ void reduce_max_final_kernel(const float* __restrict__ block_max,
                                        float* __restrict__ output,
                                        int num_blocks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_blocks) return;

    atomicMaxFloat(output, block_max[tid]);
}

extern "C" void reduce_max_launcher(const float* input,
                                    long long numel,
                                    float* output,
                                    cudaStream_t stream) {
    if (numel <= 0) return;

    const int block_size = MAX_BLOCK_SIZE;
    int max_blocks_per_grid = 1024;
    int needed_blocks = (int)ceil((double)numel / (double)block_size);
    int grid_size = min(needed_blocks, max_blocks_per_grid);

    float* block_max;
    cudaMalloc(&block_max, grid_size * sizeof(float));

    reduce_max_block_kernel<MAX_BLOCK_SIZE><<<grid_size, MAX_BLOCK_SIZE, 0, stream>>>(
        input, block_max, numel);

    // Initialize output to -INFINITY on the same stream before final reduction.
    cudaMemsetAsync(output, 0xFF, sizeof(float), stream); // 0xFF000000 is -inf for float
    // Actually need to set to -inf: use a small kernel or cudaMemsetAsync with 0xFF? For IEEE 754, -inf is 0xFF800000.
    // Better: use cudaMemsetAsync with 0xFF for all bytes gives -NaN? We'll use a temporary device variable set via cudaMemcpyAsync from a host constant, or simpler: write a small kernel.
    // For simplicity, we'll use cudaMemsetAsync with 0xFF for all bytes and then reinterpret? But 0xFFFFFFFF is NaN, not -inf. Let's use a temporary device array initialized to -INFINITY via cudaMemcpyToSymbol or a small kernel.
    // Actually, we can avoid initialization entirely by having the final kernel first check if tid==0 and do a non-atomic store to -INFINITY then __threadfence() then atomics? But that still has race. Since we have only one block, we can set output to -inf by thread0 and then __syncthreads() before atomics, but that is what was there and it didn't work. Maybe the issue is that blockIdx.x is not zero? In final kernel grid=1, block=256, so blockIdx.x=0. So thread0 sets it and __syncthreads() ensures all threads see it before proceeding. That should be safe. So why did it fail? Possibly because the output was not initialized to -inf but instead left as garbage, and the atomicMaxFloat with garbage could produce wrong result. But we set it to -inf. However, the atomicMaxFloat function may not handle -inf correctly because the atomicCAS loop might not converge if the initial value is -inf. Actually it should work. Let's inspect: old = *addr_as_int (reads -inf), loop: fmaxf(-inf, value) = value, atomicCAS(addr, -inf, value) succeeds, old becomes -inf, loop ends with old = -inf which equals assumed, exit. Fine.
    // So perhaps the issue is elsewhere: the grid-stride loop in block kernel might skip some elements if numel is large and grid is capped. Let's compute: block_size=256, grid_size=1024, stride=262144. numel=314572800. tid ranges 0..262143. Loop covers all indices up to numel-1? Starting at tid, increment by stride, yes it covers all because stride > max grid threads? Actually stride = 256*1024 = 262144. Each thread processes indices tid, tid+262144, tid+524288, ... up to < numel. Since numel=314572800, number of iterations per thread is ceil((numel - tid)/stride) which is about 1200. That's fine.
    // So maybe the atomicMaxFloat is wrong due to missing volatile or memory ordering? Let's use atomicMax on int instead, but we need to handle negative numbers correctly. We can write a specialized atomicMax for float using bit manipulation with integer atomicMax if we can convert float to a monotonic integer mapping. For IEEE 754, if sign bit is 0, integer order matches float order; if sign bit is 1, integer order is reversed. We can write a helper to convert float to a sortable int: if f >= 0, return int_as_bits; else return ~int_as_bits | 0x80000000. Then atomicMax on int works. That might be more robust.
    // But perhaps the simpler fix is to avoid two-stage reduction and use a single kernel with cooperative groups or just use a single block for the entire reduction? The tensor is huge (314M elements), so one block would be too slow but correctness first. We can implement a single-block reduction with grid-stride loop for all elements, but one block with 1024 threads still processes all elements, but it would be slow but correct. We'll fix correctness first.
    // To be safe, we'll implement a single-block reduction for max using a single kernel that has grid=1, block=MAX_BLOCK_SIZE (256), and each thread loops over the whole tensor with stride=blockDim.x. This is simple and correct. Since the tensor is 314M, each thread processes ~1.2M elements; with 256 threads, total ~314M, should be okay performance-wise (maybe not optimal but acceptable). For std, similarly we can do a single-block two-pass to avoid catastrophic cancellation.
    // We'll rewrite both operations to use a single kernel for simplicity and correctness.
