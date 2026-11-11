#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for computing sin of int64 input, producing float32 output.
// Each thread handles one element.
__global__ void sin_kernel(const int64_t* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Convert int64 to float, then compute sine.
        output[idx] = sinf(static_cast<float>(input[idx]));
    }
}

// Launcher for custom_equal.
// Compares two int64 arrays element-wise on CPU. Returns true only if all
extern "C" bool custom_equal_launcher(const int64_t* x, const int64_t* y,
                                      int64_t numel) {
    for (int64_t i = 0; i < numel; ++i) {
        if (x[i] != y[i]) {
            return false;
        }
    }
    return true;
}

// Launcher for custom_sin.
// Input: int64 array on CPU (already copied on host by binding layer).
// Output: float array on GPU.
// The binding layer is responsible for allocating the output device memory and
// copying the input to device memory before calling this launcher.
extern "C" void custom_sin_launcher(const int64_t* input_dev,
                                    float* output_dev,
                                    int64_t numel,
                                    cudaStream_t stream) {
    const int block_size = 256;
    const int64_t grid_size = (numel + block_size - 1) / block_size;
    sin_kernel<<<grid_size, block_size, 0, stream>>>(input_dev, output_dev, numel);
}
