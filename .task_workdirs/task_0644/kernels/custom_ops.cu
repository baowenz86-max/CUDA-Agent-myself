#include <cuda_runtime.h>
#include <math.h>

// Kernel for mean_dim1_keepdim: given input shape [N, 1, H, W], dim=1 length is 1,
// so the mean over dim 1 equals the input element itself. This kernel simply copies
// the input to the output (elementwise).
__global__ void mean_dim1_keepdim_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i];
    }
}

// Kernel for floor: elementwise floorf
template <typename scalar_t>
__global__ void floor_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = floorf((float)input[i]);
    }
}

// Launcher for mean_dim1_keepdim
extern "C" void mean_dim1_keepdim_launcher(const float* input,
                                            float* output,
                                            int64_t numel,
                                            cudaStream_t stream) {
    // Choose block size and grid size (grid-stride loop handles arbitrary numel)
    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    // Cap grid size to avoid excessive blocks (grid-stride loop will handle the rest)
    if (grid_size > 65535) grid_size = 65535;
    mean_dim1_keepdim_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}

// Launcher for floor
extern "C" void floor_launcher(const float* input,
                                float* output,
                                int64_t numel,
                                cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;
    floor_kernel<float><<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
