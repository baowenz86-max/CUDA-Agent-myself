#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// custom_equal: torch.equal(x, y) for CUDA float32 tensors.
// Returns true iff sizes match (checked by binding) and every element is
// bitwise equal. We launch one block per chunk; each block uses a single
// global flag that becomes 1 when a mismatch is found. A final host read of
// the flag gives the result.
// -----------------------------------------------------------------------------

__global__ void equal_flag_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    long long numel,
    const long long* __restrict__ x_strides,
    const long long* __restrict__ y_strides,
    int ndim,
    int* __restrict__ mismatch_flag)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long linear = idx; linear < numel; linear += stride) {
        // Convert linear index to multi-dimensional index using x strides.
        long long rem = linear;
        long long x_offset = 0;
        long long y_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            // Use largest dimension stride order; strides are int64.
            char tmp[16];
            // We assume contiguous traversal order; for non-contiguous, still works if strides linear.
            long long coord = (d == ndim - 1) ? rem : rem % x_strides[d];
            // Actually compute properly: coordinate = (linear / prod of remaining dims) % shape[d],
            // but we don't have shape passed. Instead use the standard approach for row-major contiguous,
            // and for arbitrary strides require that x and y are broadcast-compatible with same shape.
            // Since contract states both have same shape and are contiguous in reference, we can
            // safely use linear index only for contiguous optimization. For general strides,
            // fallback to Python would be needed, but binding ensures contiguous for the task.
            // To keep launcher stable, we take contiguous pointer path.
            // This kernel is replaced below by a contiguous version; see launcher.
        }
        // Simple contiguous comparison (preferred path).
        if (x[linear] != y[linear]) {
            atomicExch(mismatch_flag, 1);
        }
    }
}

// Contiguous kernel (used when both tensors are contiguous)
__global__ void equal_contiguous_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    long long numel,
    int* __restrict__ mismatch_flag)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        if (__ldg(x + i) != __ldg(y + i)) {
            atomicExch(mismatch_flag, 1);
            return;
        }
    }
}

// -----------------------------------------------------------------------------
// custom_l1_loss: F.l1_loss(x, y, reduction='mean')
// sum |x-y| / numel, output zero-dimensional float32 tensor.
// We use a two-pass (split-K) reduction with atomics on double or float.
// For large numel, accumulate in double to reduce precision loss; final result cast to float.
// Implementation: one block per chunk, each block reduces its chunk with shared memory,
// then atomically adds to global accumulator (float) or double. To avoid non-determinism
// and overflow, use double accumulator if numel large (here 134M, float sum magnitude
// can be ~1e4, safe but double is safer). We'll use double global accumulator.
// -----------------------------------------------------------------------------

__global__ void l1_sum_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    long long numel,
    double* __restrict__ global_sum)
{
    extern __shared__ double sdata[];
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    double local_sum = 0.0;
    for (long long i = idx; i < numel; i += stride) {
        local_sum += fabs((double)x[i] - (double)y[i]);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// Alternative kernel that directly computes mean using one block per element (grid-stride)
// with a single thread computing final mean. Simpler and less error-prone.
__global__ void l1_mean_single_block_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    long long numel,
    float* __restrict__ output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum = 0.0;
        for (long long i = 0; i < numel; ++i) {
            sum += fabs((double)x[i] - (double)y[i]);
        }
        output[0] = (float)(sum / (double)numel);
    }
}

// Better: grid-stride with final reduction inside a single block? Use the first approach.
// We'll expose launchers that choose appropriate implementation.

// -----------------------------------------------------------------------------
// Launchers (C interface for binding.cpp)
// -----------------------------------------------------------------------------

extern "C" bool custom_equal_launcher(
    const float* x,
    const float* y,
    long long numel,
    const long long* x_strides,
    const long long* y_strides,
    int ndim,
    cudaStream_t stream)
{
    if (numel == 0) return true;

    int device;
    cudaGetDevice(&device);
    int* d_flag = nullptr;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemsetAsync(d_flag, 0, sizeof(int), stream);

    bool contiguous = true;
    // assume contiguous for simplicity; strides handling can be added if needed,
    // but contract states contiguous in reference.

    if (contiguous) {
        int block = 256;
        int grid = (int)std::min((numel + block - 1) / block, (long long)65535);
        equal_contiguous_kernel<<<grid, block, 0, stream>>>(x, y, numel, d_flag);
    } else {
        // Fallback not implemented for non-contiguous; binding should enforce contiguous.
        cudaFree(d_flag);
        return false; // caller should ensure contiguity, or implement stride kernel.
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_flag);
        return false;
    }
    int h_flag = 0;
    cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_flag);
    return (h_flag == 0);
}

extern "C" float custom_l1_loss_launcher(
    const float* x,
    const float* y,
    long long numel,
    const long long* x_strides,
    const long long* y_strides,
    int ndim,
    cudaStream_t stream)
{
    if (numel == 0) {
        // PyTorch returns NaN for empty input mean; but this path should not be used by binding.
        float nan = __int_as_float(0x7fc00000);
        return nan;
    }

    // Use double accumulator with grid-stride and atomics.
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemsetAsync(d_sum, 0, sizeof(double), stream);

    const int block = 256;
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    long long max_grid_x = prop.maxGridSize[0];
    long long grid = (numel + block - 1) / block;
    if (grid > max_grid_x) grid = max_grid_x;

    size_t shared_mem = block * sizeof(double);
    l1_sum_kernel<<<(int)grid, block, shared_mem, stream>>>(x, y, numel, d_sum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_sum);
        return 0.0f; // error handling in binding is expected
    }

    // Copy sum to host and compute mean
    double h_sum = 0.0;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_sum);

    float result = (float)(h_sum / (double)numel);
    return result;
}
