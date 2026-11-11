// Custom CUDA kernels for ELU and mean reduction
#include <cuda_runtime.h>

// ELU activation kernel: out[i] = x[i] > 0 ? x[i] : expf(x[i]) - 1.0f
__global__ void elu_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : (expf(x) - 1.0f);
    }
}

// Forward declarations for kernels used in mean_all_forward_launcher
__global__ void mean_atomic_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t numel);
__global__ void mean_div_kernel(float* __restrict__ output, int64_t numel);

// Launcher for ELU forward
extern "C" void elu_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    if (numel <= 0) return; // no elements

    const int threads = 256;
    // Grid-stride loop, choose grid size to cover all SMs efficiently
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024; // cap grid to avoid excessive blocks
    elu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for mean of all elements
extern "C" void mean_all_forward_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          cudaStream_t stream) {
    if (numel <= 0) {
        // Empty input: the binding already rejects this case, but if called,
        // set output to NaN using cudaMemsetAsync with a float NaN pattern.
        // The previous version using __int_as_float from host code was invalid.
        // We avoid device intrinsic in host by writing directly to device via memset.
        float nan_val = 0.0f;
        // Use a known NaN bit pattern for float (0x7fc00000)
        // cudaMemsetAsync requires a byte value, so we cannot set arbitrary bits easily.
        // Instead, we allocate a small temporary on host and copy, but that adds complexity.
        // Since binding checks x.numel() > 0, we can simply return without doing anything.
        // But to be safe, we return; the binding's TORCH_CHECK will prevent this call.
        return;
    }

    const int threads = 256;
    // Determine number of blocks for partial reduction: enough to cover numel, but not too many.
    // We'll use a grid-stride loop so number of blocks can be capped.
    int max_blocks_per_sm = 32; // typical occupancy
    int device;
    cudaGetDevice(&device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int num_blocks = sm_count * max_blocks_per_sm;
    if (num_blocks == 0) num_blocks = 1;

    // Initialize output to zero
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    // Launch atomic reduction kernel
    const int threads_atomic = 256;
    int blocks_atomic = (int)((numel + threads_atomic - 1) / threads_atomic);
    if (blocks_atomic > 1024) blocks_atomic = 1024;
    mean_atomic_kernel<<<blocks_atomic, threads_atomic, 0, stream>>>(input, output, numel);

    // Launch division kernel
    mean_div_kernel<<<1, 1, 0, stream>>>(output, numel);
}

// Atomic mean kernel: each thread computes partial sum, then atomically adds to output
__global__ void mean_atomic_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t numel) {
    float sum = 0.0f;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        sum += input[i];
    }
    atomicAdd(output, sum);
}

// Division kernel to compute mean
__global__ void mean_div_kernel(float* __restrict__ output, int64_t numel) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output /= (float)numel;
    }
}
