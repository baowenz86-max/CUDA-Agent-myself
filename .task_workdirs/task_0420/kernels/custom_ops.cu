// custom_ops.cu - CUDA kernels for custom_equal and custom_smooth_l1_loss
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <algorithm>

// Kernel: check equality of two float arrays, write 0 to result if any mismatch, else 1
__global__ void equal_kernel(const float* __restrict__ a, const float* __restrict__ b, int64_t numel, int* __restrict__ result) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        if (a[i] != b[i]) {
            atomicExch(result, 1);
            return;
        }
    }
}

// Kernel: compute smooth L1 loss sum
__global__ void smooth_l1_loss_sum_kernel(const float* __restrict__ input, const float* __restrict__ target, int64_t numel, float* __restrict__ sum) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + tid;
    float local_sum = 0.0f;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float diff = input[i] - target[i];
        float abs_diff = fabsf(diff);
        float loss;
        if (abs_diff < 1.0f) {
            loss = 0.5f * diff * diff;
        } else {
            loss = abs_diff - 0.5f;
        }
        local_sum += loss;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

// Kernel to scale output by alpha (for mean computation)
__global__ void scale_output_kernel(float* output, float alpha) {
    *output *= alpha;
}

// Launcher for custom_equal: compares two CUDA float tensors, returns 1 if equal, 0 otherwise.
// host_result must point to pinned host memory or a pageable memory location that is valid until sync.
extern "C" void custom_equal_launcher(const float* x, const float* y, int64_t numel, cudaStream_t stream, int* host_result) {
    // Allocate device int
    int* result_device = nullptr;
    cudaError_t err = cudaMalloc(&result_device, sizeof(int));
    if (err != cudaSuccess) {
        *host_result = 1; // error indicator
        return;
    }
    cudaMemsetAsync(result_device, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    equal_kernel<<<blocks, threads, 0, stream>>>(x, y, numel, result_device);

    // Asynchronously copy result to host, use a callback or sync to ensure copy completes before freeing
    cudaMemcpyAsync(host_result, result_device, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // We cannot free result_device immediately because the copy may still be pending.
    // Use cudaFreeAsync if available, otherwise schedule a callback or leak for safety.
    // Here we use cudaFreeAsync on the same stream (requires CUDA 11.2+).
    #if CUDART_VERSION >= 11020
        cudaFreeAsync(result_device, stream);
    #else
        // Fallback: leak to avoid premature free (one small allocation per call)
        // In practice this is acceptable for this task, but ideally use a memory pool.
        ;
    #endif
}

// Launcher for custom_smooth_l1_loss: computes mean smooth l1 loss, writes float to output device pointer.
extern "C" void custom_smooth_l1_loss_launcher(const float* x, const float* y, int64_t numel, float* output, cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    int smem_size = threads * sizeof(float);
    smooth_l1_loss_sum_kernel<<<blocks, threads, smem_size, stream>>>(x, y, numel, output);
    float alpha = 1.0f / (float)numel;
    scale_output_kernel<<<1, 1, 0, stream>>>(output, alpha);
}
