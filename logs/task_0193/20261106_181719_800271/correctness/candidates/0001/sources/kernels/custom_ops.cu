#include <cuda_runtime.h>

// Kernel: compute mean along dimension 1 (channels) with keepdim=True
// Input: x [B, C, H, W] with strides sxB, sxC, sxH, sxW
// Output: out [B, 1, H, W] with strides soB, so0, soH, soW
// Each thread computes one output element (b, h, w).
__global__ void mean_dim1_keepdim_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t sxB, int64_t sxC, int64_t sxH, int64_t sxW,
    int64_t soB, int64_t so0, int64_t soH, int64_t soW)
{
    int64_t total = B * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int64_t b = idx / (H * W);
        int64_t rem = idx % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        const float* x_base = x + b * sxB + h * sxH + w * sxW;
        double sum = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            sum += (double)x_base[c * sxC];
        }
        float mean = (float)(sum / (double)C);

        float* out_ptr = out + b * soB + h * soH + w * soW;
        *out_ptr = mean;
    }
}

extern "C" void custom_mean_dim1_keepdim_launcher(
    float* output,
    const float* input,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t sxB, int64_t sxC, int64_t sxH, int64_t sxW,
    int64_t soB, int64_t so0, int64_t soH, int64_t soW,
    cudaStream_t stream)
{
    if (B == 0 || H == 0 || W == 0 || C == 0) {
        // no elements to compute, return early
        return;
    }

    int64_t total = B * H * W;
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // cap grid to avoid too many blocks (grid-stride loop handles rest)
    const int64_t max_grid_y = 65535;
    if (grid_size > max_grid_y) {
        grid_size = max_grid_y;
    }

    mean_dim1_keepdim_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, B, C, H, W, sxB, sxC, sxH, sxW, soB, so0, soH, soW);
}
