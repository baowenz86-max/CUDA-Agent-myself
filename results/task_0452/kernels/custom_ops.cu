#include <cuda_runtime.h>

// Elementwise sigmoid: y = 1 / (1 + exp(-x))
__global__ void sigmoid_kernel(const float* __restrict__ x,
                              float* __restrict__ y,
                              int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// Elementwise ReLU: y = max(0, x)
__global__ void relu_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

// Add bias broadcast: y[n,c,h,w] = x[n,c,h,w] + bias[c]
// Assumes x is contiguous with shape [N, C, H, W], bias contiguous length C.
__global__ void add_bias_kernel(const float* __restrict__ x,
                                const float* __restrict__ bias,
                                float* __restrict__ y,
                                int64_t total,
                                int64_t C,
                                int64_t HW) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        // channel index = (i / HW) % C
        int64_t c = (i / HW) % C;
        y[i] = x[i] + bias[c];
    }
}

extern "C" void custom_sigmoid_launcher(const float* x,
                                       float* y,
                                       int64_t total,
                                       cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding device limits (we use grid-stride loop)
    int max_blocks = 65535;  // safe for all CUDA devices
    if (grid_size > max_blocks) grid_size = max_blocks;
    sigmoid_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, total);
}

extern "C" void custom_relu_launcher(const float* x,
                                     float* y,
                                     int64_t total,
                                     cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    int max_blocks = 65535;
    if (grid_size > max_blocks) grid_size = max_blocks;
    relu_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, total);
}

extern "C" void custom_add_bias_launcher(const float* x,
                                         const float* bias,
                                         float* y,
                                         int64_t total,
                                         int64_t C,
                                         int64_t HW,
                                         cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    int max_blocks = 65535;
    if (grid_size > max_blocks) grid_size = max_blocks;
    add_bias_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, bias, y, total, C, HW);
}
