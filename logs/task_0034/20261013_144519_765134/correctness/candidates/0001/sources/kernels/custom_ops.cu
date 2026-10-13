#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Nearest neighbor upsample (2D) kernel
// Input:  float, shape [N, C, H, W]
// Output: float, shape [N, C, H*scale, W*scale]
// scale is an integer factor (typically 2)
// -----------------------------------------------------------------------------
__global__ void upsample_nearest2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int scale) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * H * W * scale * scale;
    if (out_idx >= total_out) return;

    // Decompose output index: n, c, out_h, out_w
    int out_w = out_idx % (W * scale);
    int tmp = out_idx / (W * scale);
    int out_h = tmp % (H * scale);
    tmp = tmp / (H * scale);
    int c = tmp % C;
    int n = tmp / C;

    int in_h = out_h / scale;
    int in_w = out_w / scale;

    int in_idx = ((n * C + c) * H + in_h) * W + in_w;
    output[out_idx] = input[in_idx];
}

extern "C" void upsample_nearest2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale,
    cudaStream_t stream) {

    int total_out = N * C * H * W * scale * scale;
    const int block = 256;
    int grid = (total_out + block - 1) / block;
    upsample_nearest2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W, scale);
}

// -----------------------------------------------------------------------------
// One-hot kernel
// Input:  int64_t, shape [N, H, W]
// Output: int64_t, shape [N, H, W, num_classes]
// Values outside [0, num_classes-1] produce all-zero rows (PyTorch behavior)
// -----------------------------------------------------------------------------
__global__ void one_hot_kernel(
    const int64_t* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t total_spatial,  // N*H*W
    int64_t num_classes) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_spatial) return;

    int64_t label = input[idx];
    int64_t out_base = idx * num_classes;

    // Set all classes to zero first
    for (int64_t j = 0; j < num_classes; ++j) {
        output[out_base + j] = 0;
    }

    if (label >= 0 && label < num_classes) {
        output[out_base + label] = 1;
    }
}

extern "C" void one_hot_launcher(
    const int64_t* input,
    int64_t* output,
    int64_t total_spatial,
    int64_t num_classes,
    cudaStream_t stream) {

    const int block = 256;
    int64_t grid = (total_spatial + block - 1) / block;
    one_hot_kernel<<<grid, block, 0, stream>>>(
        input, output, total_spatial, num_classes);
}
