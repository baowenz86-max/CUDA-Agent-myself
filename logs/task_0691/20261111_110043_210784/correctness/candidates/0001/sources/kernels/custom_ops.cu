#include <cuda_runtime.h>
#include <cstdio>

// MSE sum reduction kernel
// Computes partial sums of (x - y)^2 for each block.
// Each block uses shared memory reduction.
__global__ void mse_sum_kernel_partial(float* __restrict__ partial_sums,
                                       const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       long n) {
    extern __shared__ float sdata[];
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = gridDim.x * blockDim.x;
    float local_sum = 0.0f;

    // Grid-stride loop for all elements
    for (long i = tid; i < n; i += stride) {
        float diff = x[i] - y[i];
        local_sum += diff * diff;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel: sums all partial sums into a single value.
__global__ void mse_sum_finalize(float* __restrict__ out,
                                 const float* __restrict__ partial_sums,
                                 int num_partials) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local_sum = 0.0f;

    // Load partial sums into shared memory (each block handles a chunk if needed)
    for (int i = tid; i < num_partials; i += blockDim.x) {
        local_sum += partial_sums[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}

// Helper to launch kernel and check errors
static void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        // In a real extension we would throw, but here we just report.
    }
}

#define CUDA_CHECK(call) checkCudaError((call), __FILE__, __LINE__)

// Launcher function (no torch dependencies)
extern "C" void mse_sum_launcher(float* out,
                                 const float* x,
                                 const float* y,
                                 long n,
                                 cudaStream_t stream) {
    if (n <= 0) {
        // Empty input: mean is undefined, but we set out to 0 to avoid NaN.
        // The binding layer should handle empty tensors.
        cudaMemsetAsync(out, 0, sizeof(float), stream);
        return;
    }

    // Choose block size and grid size.
    int block_size = 256;
    // Maximum number of blocks is limited, but we use grid-stride loop, so we can use
    // a reasonable number of blocks that covers the whole SM count.
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    // Use enough blocks to keep all SMs busy, but not more than necessary.
    // Typical choice: 4 * sm_count. However we also need at least ceil(n / block_size)
    // blocks if n is small. We'll compute min(4*sm_count, (n + block_size - 1) / block_size).
    long max_blocks = (n + block_size - 1) / block_size;
    int grid_size = (int)min((long)(4 * sm_count), max_blocks);
    if (grid_size < 1) grid_size = 1;

    // Allocate temporary buffer for partial sums on device.
    // We'll allocate using cudaMallocAsync if available, else cudaMalloc.
    // For simplicity, use cudaMalloc and cudaFree, but the binding should pass a preallocated buffer?
    // Since the launcher must be self-contained, we allocate here.
    // This may be a performance bottleneck if called repeatedly; but the task did not forbid it.
    // To avoid repeated allocations, we can use a static device buffer or rely on the binding to provide one.
    // Contract says launcher should use raw pointers and scalars only, but we can still allocate.
    // We'll allocate a workspace large enough for grid_size floats.
    float* d_partial = nullptr;
    cudaError_t err = cudaMalloc(&d_partial, grid_size * sizeof(float));
    CUDA_CHECK(err);
    if (err != cudaSuccess) {
        return;
    }

    // Launch partial sum kernel
    mse_sum_kernel_partial<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        d_partial, x, y, n);
    CUDA_CHECK(cudaGetLastError());

    // Launch final reduction kernel (single block)
    int final_block_size = 256; // can be smaller if grid_size < block_size
    if (grid_size < final_block_size) final_block_size = grid_size;
    int final_shared_mem = final_block_size * sizeof(float);
    mse_sum_finalize<<<1, final_block_size, final_shared_mem, stream>>>(
        out, d_partial, grid_size);
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffer after kernel completion (asynchronously?)
    // We can use cudaFree, which synchronizes? Actually cudaFree is synchronous-ish.
    // Better to use cudaFree, but it may block. For simplicity, we just free.
    cudaFree(d_partial);
}
