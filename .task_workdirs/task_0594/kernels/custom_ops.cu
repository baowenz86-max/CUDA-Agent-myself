#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// custom_cat: Concatenate two float32 CUDA tensors along dimension 1.
// Inputs are expected to be 4D contiguous tensors with shapes
// [batch, C, H, W] and same batch, H, W; dim is the concatenation dimension
// (here dim=1). Output is a contiguous float32 tensor with concatenated
// dimension size C1+C2.
//
// The launcher assumes:
//   - x1, x2, out are valid CUDA device pointers
//   - all tensors are float32
//   - shapes are [batch, C1, H, W] and [batch, C2, H, W]
//   - dim == 1 (other dims not implemented per contract)
//   - all inputs are contiguous (contract says supports strides, but this
//     implementation relies on contiguous; binding layer may call .contiguous())
// -----------------------------------------------------------------------------

// Simple copy kernel: copies a block of data from src to dst.
template <typename T>
__global__ void copy_kernel(const T* __restrict__ src,
                            T* __restrict__ dst,
                            int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        dst[i] = src[i];
    }
}

// Kernel that concatenates two tensors along dimension 1 for 4D layout.
// Each thread handles one contiguous slice of size (C1*H*W) or (C2*H*W)
// within a batch. This is efficient because the inner dimensions are contiguous.
template <typename T>
__global__ void concat_dim1_kernel(const T* __restrict__ x1,
                                   const T* __restrict__ x2,
                                   T* __restrict__ out,
                                   int64_t batch,
                                   int64_t C1,
                                   int64_t C2,
                                   int64_t H,
                                   int64_t W) {
    int64_t plane_size = H * W;
    int64_t slice1_size = C1 * plane_size;
    int64_t slice2_size = C2 * plane_size;
    int64_t total_slices = batch;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t b = idx; b < total_slices; b += stride) {
        const T* src1 = x1 + b * slice1_size;
        const T* src2 = x2 + b * slice2_size;
        T* dst = out + b * (slice1_size + slice2_size);

        // Copy first tensor's slice
        for (int64_t i = 0; i < slice1_size; ++i) {
            dst[i] = src1[i];
        }
        // Copy second tensor's slice
        for (int64_t i = 0; i < slice2_size; ++i) {
            dst[slice1_size + i] = src2[i];
        }
    }
}

// Launcher for custom_cat with dim=1 and float32.
// Parameters:
//   out:       preallocated output tensor of shape [batch, C1+C2, H, W]
//   x1:        first input [batch, C1, H, W]
//   x2:        second input [batch, C2, H, W]
//   batch, C1, C2, H, W: dimensions
//   stream:    CUDA stream to launch on
extern "C" void custom_cat_launcher(float* out,
                                    const float* x1,
                                    const float* x2,
                                    int64_t batch,
                                    int64_t C1,
                                    int64_t C2,
                                    int64_t H,
                                    int64_t W,
                                    cudaStream_t stream) {
    int threads = 256;
    int blocks = 1024; // grid-stride loop handles large batch

    // If total number of elements is small, adjust blocks
    int64_t total_elements = batch * (C1 + C2) * H * W;
    if (total_elements < (int64_t)threads * blocks) {
        blocks = (int)((total_elements + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
    }

    // Use concat_dim1_kernel for performance; fallback to naive copy if
    // dimensions are strange (not expected). Here always use specialized kernel.
    concat_dim1_kernel<float><<<blocks, threads, 0, stream>>>(
        x1, x2, out, batch, C1, C2, H, W);
}

// -----------------------------------------------------------------------------
// custom_equal: Compare two float32 tensors element-wise and return whether
// all elements are exactly equal. Returns a bool (as int) via out_equal pointer.
//
// The launcher assumes:
//   - x1, x2 are valid CUDA device pointers
//   - both have the same numel and are float32
//   - out_equal is a device pointer to int (0 or 1)
// -----------------------------------------------------------------------------

// Kernel that computes per-block equality and atomically combines into global result.
template <typename T>
__global__ void equal_atomic_kernel(const T* __restrict__ x1,
                                    const T* __restrict__ x2,
                                    int* __restrict__ result,
                                    int64_t numel) {
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    int64_t stride = blockDim.x * gridDim.x;

    int local_equal = 1;
    for (int64_t i = idx; i < numel; i += stride) {
        if (x1[i] != x2[i]) {
            local_equal = 0;
            break;
        }
    }

    // Combine within block using shared memory
    __shared__ int s_equal[256]; // blockDim.x assumed 256
    s_equal[tid] = local_equal;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_equal[tid] &= s_equal[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAnd(result, s_equal[0]);
    }
}

// Launcher for custom_equal using atomicAnd.
extern "C" void custom_equal_launcher(const float* x1,
                                      const float* x2,
                                      int* result,
                                      int64_t numel,
                                      cudaStream_t stream) {
    if (numel == 0) {
        cudaMemsetAsync(result, 1, sizeof(int), stream);
        return;
    }

    int threads = 256;
    int max_blocks = 65535; // max grid dim
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > max_blocks) blocks = max_blocks;

    // result must be initialized to 1 before calling this launcher.
    // Binding layer is responsible for cudaMemsetAsync(result, 1, ...).

    equal_atomic_kernel<float><<<blocks, threads, 0, stream>>>(
        x1, x2, result, numel);
}
