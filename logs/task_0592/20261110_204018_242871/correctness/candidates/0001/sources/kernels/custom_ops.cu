// custom_ops.cu - CUDA kernels for cat and ceil
#include <cuda_runtime.h>
#include <math.h>

// Kernel to copy data from input1 and input2 into output, assuming dim=0 semantics
// For dim=0, concatenation is just sequential memory copy of contiguous tensors.
// This kernel works for any concatenation along the first dimension only.
// If dim != 0, this kernel would need to handle non-contiguous slices; but
// the contract specifies dim=0, so we implement the generic case with a check
// inside the launcher to handle only dim=0 for correctness.
__global__ void cat_copy_kernel(const float* __restrict__ x1,
                                const float* __restrict__ x2,
                                float* __restrict__ out,
                                long long offset2,
                                long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        if (i < offset2) {
            out[i] = x1[i];
        } else {
            out[i] = x2[i - offset2];
        }
    }
}

// Kernel for ceil on float32
__global__ void ceil_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        output[i] = ceilf(input[i]);
    }
}

// Launcher for cat_forward: concatenates x1 and x2 along dimension dim.
// Currently only supports dim=0 for contiguous float32 tensors.
// Returns 0 on success, -1 on unsupported dim or CUDA error.
extern "C" int cat_forward_launcher(const float* x1,
                                    const float* x2,
                                    float* output,
                                    long long x1_elements,
                                    long long x2_elements,
                                    int dim,
                                    cudaStream_t stream) {
    // dim must be 0 for this simple implementation.
    if (dim != 0) {
        return -1;
    }
    long long total = x1_elements + x2_elements;
    if (total == 0) {
        // nothing to do
        return 0;
    }
    int block_size = 256;
    // Use 64-bit grid size to avoid overflow for large tensors.
    long long grid_size = (total + block_size - 1) / block_size;
    // Cap grid at 2^31-1 blocks (max grid dim x).
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    cat_copy_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        x1, x2, output, x1_elements, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -2;
    }
    return 0;
}

// Launcher for ceil_forward.
extern "C" int ceil_forward_launcher(const float* input,
                                     float* output,
                                     long long n,
                                     cudaStream_t stream) {
    if (n <= 0) {
        return 0;
    }
    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    ceil_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }
    return 0;
}
