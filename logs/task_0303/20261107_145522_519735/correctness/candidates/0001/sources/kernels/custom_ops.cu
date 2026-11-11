#include <cuda_runtime.h>

// CUDA kernel for pixel_shuffle (also known as depth-to-space in some frameworks)
// Input:  x (N, C, H, W) where C = C_out * r * r
// Output: y (N, C_out, H * r, W * r)
// The mapping is:
//   y[n, c_out, h * r + i, w * r + j] = x[n, c_out * (r*r) + i * r + j, h, w]
// where 0 <= i,j < r, 0 <= h < H, 0 <= w < W
// The kernel is launched with enough blocks to cover all output elements.
// Grid-stride loop is used for safety with large tensors.

template <typename scalar_t>
__global__ void pixel_shuffle_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int r
) {
    int total_elements = N * (C / (r * r)) * (H * r) * (W * r);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        // Decode output index: (n, c_out, oh, ow)
        int tmp = idx;
        int ow = tmp % (W * r);
        tmp /= (W * r);
        int oh = tmp % (H * r);
        tmp /= (H * r);
        int c_out = tmp % (C / (r * r));
        tmp /= (C / (r * r));
        int n = tmp;

        // Compute input coordinates
        int h = oh / r;
        int w = ow / r;
        int i = oh % r;
        int j = ow % r;
        int c = c_out * (r * r) + i * r + j;

        // Input linear index (assuming contiguous row-major layout)
        int input_idx = ((n * C + c) * H + h) * W + w;
        output[idx] = input[input_idx];
    }
}

// Declare launcher with C linkage for easy binding from .cpp files
extern "C" void pixel_shuffle_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale_factor,
    cudaStream_t stream
) {
    const int block_size = 256;
    int total_elements = N * (C / (scale_factor * scale_factor)) * (H * scale_factor) * (W * scale_factor);
    int blocks = (total_elements + block_size - 1) / block_size;

    // Launch kernel
    pixel_shuffle_kernel<float><<<blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W, scale_factor
    );
}
