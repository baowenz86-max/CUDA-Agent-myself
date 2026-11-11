#include <cuda_runtime.h>
#include <stdint.h>

// Element-wise absolute value kernel for float32 input
__global__ void abs_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Unsqueeze kernel: simply copies input to output assuming output is contiguous
// and the new dimension size is 1 (so element order is unchanged)
__global__ void unsqueeze_copy_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i];
    }
}

// Launcher for abs_forward
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit for simple kernel; grid-stride handles rest
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error in binding layer; here just ignore but could record
    }
}

// Launcher for unsqueeze_forward
extern "C" void unsqueeze_forward_launcher(const float* input,
                                           float* output,
                                           int64_t numel,
                                           int64_t dim,
                                           cudaStream_t stream) {
    // dim is not used in kernel because the unsqueeze operation is just a view
    // but we need to copy data. For dim between 0 and 3, output shape [512,1,1024,1024].
    // Since output is contiguous, we can copy as flat array.
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    unsqueeze_copy_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore for now
    }
}
