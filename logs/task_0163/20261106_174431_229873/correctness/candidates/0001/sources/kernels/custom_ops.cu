#include <cuda_runtime.h>
#include <cmath>

__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                long long numel) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float x = input[i];
        // PyTorch softplus: for x > threshold (20), output = x;
        // otherwise log1p(exp(x)). Using float precision as input/output are float32.
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

extern "C" void softplus_forward_launcher(const float* input,
                                          float* output,
                                          long long numel,
                                          cudaStream_t stream) {
    const int block_size = 256;
    // Calculate grid size, capped at 65535 (max grid dimension in x).
    // Grid-stride loop handles any larger sizes.
    long long grid_size_ll = (numel + block_size - 1) / block_size;
    int grid_size = (int)min(grid_size_ll, (long long)65535);
    if (grid_size == 0) grid_size = 1; // at least one block, even if numel==0 (kernel will exit immediately)
    softplus_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
