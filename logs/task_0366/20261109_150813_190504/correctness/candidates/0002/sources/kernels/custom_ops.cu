#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// CPU comparison: torch.equal for int64 tensors
extern "C" bool custom_equal_cpu_int64_launcher(
    const int64_t* x,
    const int64_t* y,
    int64_t numel,
    cudaStream_t stream)
{
    // This is a CPU-only operation; stream is unused.
    for (int64_t i = 0; i < numel; ++i) {
        if (x[i] != y[i]) {
            return false;
        }
    }
    return true;
}

// CUDA kernel for sin(int64_t) -> float32
__global__ void sin_int64_to_float_kernel(
    const int64_t* __restrict__ input,
    float* __restrict__ output,
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = static_cast<float>(sin(static_cast<double>(input[i])));
    }
}

// Launcher for sin: copies int64 CPU tensor to GPU, computes sin, returns float32 output
extern "C" bool custom_sin_cuda_int64_launcher(
    const int64_t* cpu_input,
    int64_t numel,
    float* cuda_output,
    cudaStream_t stream)
{
    int64_t* d_input = nullptr;
    cudaError_t err = cudaMalloc(&d_input, numel * sizeof(int64_t));
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMemcpyAsync(d_input, cpu_input, numel * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        return false;
    }

    // Launch configuration: use 512 threads per block, enough blocks to cover numel with grid-stride loop
    const int block_size = 512;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Limit grid size to avoid exceeding max grid dimension (2^31-1)
    const int64_t max_grid_size = 2147483647LL;
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }

    sin_int64_to_float_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        d_input,
        cuda_output,
        numel);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        return false;
    }

    err = cudaFree(d_input);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}
