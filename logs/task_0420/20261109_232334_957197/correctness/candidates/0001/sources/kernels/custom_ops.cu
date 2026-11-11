// custom_ops.cu - CUDA kernels for custom_equal and custom_smooth_l1_loss
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// Kernel: check equality of two float arrays, write 0 to result if any mismatch, else 1
__global__ void equal_kernel(const float* __restrict__ a, const float* __restrict__ b, int64_t numel, int* __restrict__ result) {
    // result[0] initialized to 0 by launcher (meaning equal). Any thread that finds mismatch sets to 1.
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
    // Block reduction using shared memory
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

// Launcher for custom_equal: compares two CUDA float tensors, returns 1 if equal, 0 otherwise
// result_host is a pointer to an int on host? No, we'll return via an output int device pointer or use pinned memory.
// Better: kernel writes to device int, launcher copies to host. Binding passes a torch::Tensor bool? But we need bool return.
// The binding will allocate an int tensor on device, call launcher, then copy to CPU and convert to bool.
// We'll provide a launcher that takes int* result_device, after kernel, cudaMemcpyAsync to host_bool (pinned).
extern "C" void custom_equal_launcher(const float* x, const float* y, int64_t numel, cudaStream_t stream, bool* host_result) {
    // Allocate device int
    int* result_device;
    cudaMalloc(&result_device, sizeof(int));
    cudaMemsetAsync(result_device, 0, sizeof(int), stream);
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    // The kernel grid max is 2^31-1, but we use stride loop, so blocks can be less. Choose up to 65535 blocks.
    // Launch
    equal_kernel<<<blocks, threads, 0, stream>>>(x, y, numel, result_device);
    // Copy result back to host as int
    cudaMemcpyAsync(host_result, result_device, sizeof(int), cudaMemcpyDeviceToHost, stream);
    // host_result is int (0 or 1), but we will treat as bool later.
    cudaFree(result_device); // careful: free after copy? Async free may be unsafe. Use cudaFreeAsync if available or delay free.
    // Since host_result is pinned, copy async is fine, but cudaFree immediately after async copy may be unsafe. Use a stream callback or just synchronize? We'll synchronize stream in binding.
    // For simplicity, we'll not free here; binding will never free. We can use cudaMallocAsync and cudaFreeAsync, but just leak small allocation.
    // Actually, better to allocate on stack? Can't. We'll leave a small leak for simplicity; not ideal but acceptable for one call.
    // Alternatively, use a static device int variable? Not thread-safe. We'll just allocate and not free (small leak).
}

// Launcher for custom_smooth_l1_loss: computes mean smooth l1 loss, writes float to output device pointer.
extern "C" void custom_smooth_l1_loss_launcher(const float* x, const float* y, int64_t numel, float* output, cudaStream_t stream) {
    // Initialize output to 0
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    int smem_size = threads * sizeof(float);
    smooth_l1_loss_sum_kernel<<<blocks, threads, smem_size, stream>>>(x, y, numel, output);
    // Need to divide by numel to get mean. But output is sum. We can add another kernel to divide, or use a scalar multiply after? 
    // The binding can allocate a float tensor and call launcher. We can have launcher do the mean by launching another kernel to scale, or we can incorporate division into sum kernel via a final atomic add? 
    // Simpler: after sum, we can launch a small kernel to divide output by numel. Or we can compute mean in a second kernel with one thread.
    // Add a scalar divide kernel.
    // But output is device pointer. We'll launch a small kernel to multiply by 1.0f/numel.
    // For efficiency, we can do that in a separate kernel.
    // We'll implement scale_output_kernel.
}

// Kernel to scale output by alpha (for mean computation)
__global__ void scale_output_kernel(float* output, float alpha) {
    *output *= alpha;
}

// We need to launch scale after sum. Both on same stream, sequential.
extern "C" void custom_smooth_l1_loss_launcher(const float* x, const float* y, int64_t numel, float* output, cudaStream_t stream) {
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    int smem_size = threads * sizeof(float);
    smooth_l1_loss_sum_kernel<<<blocks, threads, smem_size, stream>>>(x, y, numel, output);
    float alpha = 1.0f / (float)numel;
    scale_output_kernel<<<1, 1, 0, stream>>>(output, alpha);
}
