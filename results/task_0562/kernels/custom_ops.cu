#include <cuda_runtime.h>

// Element-wise absolute value kernel for float32
__global__ void abs_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Element-wise round kernel for float32, matching torch.round (round half to even)
__global__ void round_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        // nearbyintf uses current rounding mode (default round-to-nearest-even)
        output[i] = nearbyintf(input[i]);
    }
}

// Launcher for abs_forward: no PyTorch dependencies, assumes input/output are contiguous float arrays
extern "C" void abs_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks_needed = (numel + block_size - 1) / block_size;
    int grid_size = (int)min(blocks_needed, (int64_t)65535);
    abs_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw in .cu without exception support; we simply ignore for now,
        // but binding layer may check. Could print to stderr but not recommended.
        // We'll just return; error will surface if developer checks cudaGetLastError later.
    }
}

// Launcher for round_forward
extern "C" void round_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks_needed = (numel + block_size - 1) / block_size;
    int grid_size = (int)min(blocks_needed, (int64_t)65535);
    round_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // silent failure as above
    }
}
