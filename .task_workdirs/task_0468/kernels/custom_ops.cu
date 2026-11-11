#include <cuda_runtime.h>
#include <cstdint>

// Element-wise Hardtanh (clamp) kernel for contiguous tensors.
__global__ void hardtanh_contiguous_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float min_val,
    float max_val)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, min_val), max_val);
        output[i] = val;
    }
}

// Launcher for contiguous tensors.
extern "C" void hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t blocks64 = (numel + block_size - 1) / block_size;
    const int64_t max_blocks = 65535;
    int grid_size = (int)min(blocks64, max_blocks);

    hardtanh_contiguous_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, numel, min_val, max_val);
}
