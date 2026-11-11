#include <cuda_runtime.h>
#include <device_functions.h>

// -----------------------------------------------------------------------------
// clamp_log2_forward: y = log2(clamp(x, min, +inf))
// Supports arbitrary strides via linear index and offset computation.
// Input x is float32, output y is float32, min is float.
// -----------------------------------------------------------------------------
__global__ void clamp_log2_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int64_t numel,
                                  int64_t ndim,
                                  const int64_t* __restrict__ shape,
                                  const int64_t* __restrict__ strides,
                                  float min_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Convert linear index to multidimensional indices (row-major)
    int64_t rem = idx;
    int64_t src_offset = 0;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        int64_t coord = rem % shape[d];
        rem /= shape[d];
        src_offset += coord * strides[d];
    }

    float v = x[src_offset];
    v = fmaxf(v, min_val);
    y[idx] = log2f(v);
}

// -----------------------------------------------------------------------------
// mse_loss_forward: reduce mean((x - target)^2) -> scalar
// Two-phase: each block computes partial sum, then final reduction in a separate kernel.
// -----------------------------------------------------------------------------
__global__ void mse_loss_block_sum_kernel(const float* __restrict__ x,
                                          const float* __restrict__ target,
                                          float* __restrict__ partial_sums,
                                          int64_t numel,
                                          int64_t ndim,
                                          const int64_t* __restrict__ x_shape,
                                          const int64_t* __restrict__ x_strides,
                                          const int64_t* __restrict__ target_shape,
                                          const int64_t* __restrict__ target_strides) {
    extern __shared__ float sdata[];
    int64_t tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    // Grid-stride loop to handle arbitrary numel
    for (int64_t i = idx; i < numel; i += gridDim.x * blockDim.x) {
        // Compute source offsets for x and target (same logical shape)
        int64_t rem = i;
        int64_t x_offset = 0;
        int64_t target_offset = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % x_shape[d];
            rem /= x_shape[d];
            x_offset += coord * x_strides[d];
            target_offset += coord * target_strides[d];
        }
        float diff = x[x_offset] - target[target_offset];
        local_sum += diff * diff;
    }

    // Block reduction
    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void mse_loss_final_reduce_kernel(const float* __restrict__ partial_sums,
                                             float* __restrict__ output,
                                             int num_blocks,
                                             int64_t numel) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0] / (float)numel;
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" for C++ binding)
// -----------------------------------------------------------------------------

// clamp_log2_forward
// y is output tensor (contiguous), x is input tensor (may have strides),
// shape and strides arrays are on device? No, shape and strides are small and passed as raw pointers to host memory? 
// For simplicity, we pass shape and strides as device pointers allocated in binding, but here launcher assumes they are already on device.
// But binding will pass pointers to device memory containing shape/strides. Alternatively, we can pass them as host arrays and copy to constant memory, but 
// to keep launcher simple, we accept device pointers. Binding will allocate and copy.
extern "C" void clamp_log2_launcher(const float* x,
                                    float* y,
                                    int64_t numel,
                                    int64_t ndim,
                                    const int64_t* shape,
                                    const int64_t* strides,
                                    float min_val,
                                    cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    clamp_log2_kernel<<<blocks, threads, 0, stream>>>(x, y, numel, ndim, shape, strides, min_val);
}

// mse_loss_forward
// x and target may have strides, shape arrays as device pointers.
extern "C" void mse_loss_launcher(const float* x,
                                  const float* target,
                                  float* output, // scalar output (single float)
                                  int64_t numel,
                                  int64_t ndim,
                                  const int64_t* x_shape,
                                  const int64_t* x_strides,
                                  const int64_t* target_shape,
                                  const int64_t* target_strides,
                                  cudaStream_t stream) {
    // First phase: compute block partial sums
    int threads = 256;
    int max_blocks = 1024; // reasonable max; if numel is huge, we'll use grid-stride in block_sum kernel and then final reduce handles many blocks
    int blocks = (int)min((int64_t)max_blocks, (numel + threads - 1) / threads);
    // Allocate partial_sums on device via cudaMalloc (since no torch)
    float* partial_sums;
    cudaMalloc(&partial_sums, blocks * sizeof(float));

    // Launch block sum kernel with shared memory for reduction
    size_t shared_mem = threads * sizeof(float);
    mse_loss_block_sum_kernel<<<blocks, threads, shared_mem, stream>>>(
        x, target, partial_sums, numel, ndim,
        x_shape, x_strides, target_shape, target_strides);

    // Second phase: reduce partial_sums to scalar
    int final_threads = 256;
    int final_blocks = 1;
    size_t final_shared_mem = final_threads * sizeof(float);
    mse_loss_final_reduce_kernel<<<final_blocks, final_threads, final_shared_mem, stream>>>(
        partial_sums, output, blocks, numel);

    // Cleanup partial_sums after kernel completes (asynchronous)
    cudaFreeAsync(partial_sums, stream);
}

// Note: conv_transpose1d_forward is handled by cuDNN in C++ binding; no CUDA kernel here.
// Provide an empty launcher or just declaration? The contract expects it to be implemented in binding.
// We don't need to define it here.
