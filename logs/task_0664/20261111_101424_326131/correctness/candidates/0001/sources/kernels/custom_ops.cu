#include <cuda_runtime.h>

// Kernel: compute mean along dimension 1 (channels).
// Input shape: [N, C, H, W], output shape: [N, H, W].
// Each thread computes one output element (n, h, w) by reducing over C.
// Input may be non-contiguous; use provided strides.
__global__ void mean_dim_1_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int H, int W,
                                 int stride_n, int stride_c, int stride_h, int stride_w) {
    // 2D grid: blockIdx.x covers h*w (flattened spatial), blockIdx.y covers batch n
    int spatial_index = blockIdx.x * blockDim.x + threadIdx.x; // index in [0, H*W)
    int n = blockIdx.y;
    if (n >= N || spatial_index >= H * W) return;

    int h = spatial_index / W;
    int w = spatial_index - h * W;

    const float* base = input + n * stride_n + h * stride_h + w * stride_w;
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += base[c * stride_c];
    }
    output[n * (H * W) + spatial_index] = sum / C;
}

// Launcher: host C-interface, no PyTorch dependencies.
// Args:
//   input       - pointer to input data (float)
//   output      - pointer to output data (float, preallocated)
//   N, C, H, W  - dimensions
//   stride_n, stride_c, stride_h, stride_w - strides in elements (not bytes)
//   stream      - CUDA stream
// Returns: cudaError_t status.
extern "C" cudaError_t mean_dim_1_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           int stride_n, int stride_c, int stride_h, int stride_w,
                                           cudaStream_t stream) {
    if (N == 0 || C == 0 || H == 0 || W == 0) {
        // No work; return success.
        return cudaSuccess;
    }

    int spatial_size = H * W;
    int block_size = 256;
    int grid_x = (spatial_size + block_size - 1) / block_size;
    int grid_y = N;
    dim3 grid(grid_x, grid_y);
    dim3 block(block_size);

    mean_dim_1_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w);

    // Return error from launch (caller should also cudaGetLastError)
    return cudaGetLastError();
}
