#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// reduce_mean_dim1
// Computes mean over dimension 1 (channels) with keepdim=True.
// Input x: shape [N, C, H, W], arbitrary strides.
// Output y: shape [N, 1, H, W], arbitrary strides.
// Each thread processes one output element (n, h, w) and sequentially sums
// over C. This matches torch.mean sequential accumulation order over C.
// -----------------------------------------------------------------------------
__global__ void reduce_mean_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    int64_t total_output)  // N * H * W
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_output; linear_idx += grid_size) {
        // Decompose linear index into (n, h, w) assuming H*W contiguous flattening
        int64_t n = linear_idx / (H * W);
        int64_t rem = linear_idx % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        float sum = 0.0f;
        const float* x_base = x + n * stride_x_n + h * stride_x_h + w * stride_x_w;

        #pragma unroll 4
        for (int64_t c = 0; c < C; ++c) {
            sum += x_base[c * stride_x_c];
        }

        y[n * stride_y_n + 0 * stride_y_c + h * stride_y_h + w * stride_y_w] = sum / (float)C;
    }
}

// -----------------------------------------------------------------------------
// mul_scalar
// Element-wise multiplication of x by scalar, arbitrary strides.
// -----------------------------------------------------------------------------
__global__ void mul_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float scalar,
    int64_t total_elements,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += grid_size) {
        // Decompose linear index into (n,c,h,w)
        int64_t n = linear_idx / (C * H * W);
        int64_t rem = linear_idx % (C * H * W);
        int64_t c = rem / (H * W);
        rem = rem % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        float val = x[n * stride_x_n + c * stride_x_c + h * stride_x_h + w * stride_x_w];
        y[n * stride_y_n + c * stride_y_c + h * stride_y_h + w * stride_y_w] = val * scalar;
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern C for binding layer)
// -----------------------------------------------------------------------------
extern "C" void reduce_mean_dim1_launcher(
    const float* x,
    float* y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream)
{
    int64_t total_output = N * H * W;
    if (total_output == 0) return;

    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    // Cap blocks to avoid excessive grid size; kernel uses grid-stride loop
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    reduce_mean_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, y,
        N, C, H, W,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        total_output);
}

extern "C" void mul_scalar_launcher(
    const float* x,
    float* y,
    float scalar,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream)
{
    int64_t total_elements = N * C * H * W;
    if (total_elements == 0) return;

    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    mul_scalar_kernel<<<blocks, threads, 0, stream>>>(
        x, y, scalar, total_elements,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        N, C, H, W);
}
