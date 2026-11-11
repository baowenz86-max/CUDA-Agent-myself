#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// Kernel for max over dim=1, keepdim=True
// Input: x [B, C, H, W] with strides (sx_b, sx_c, sx_h, sx_w)
// Output: values [B, 1, H, W], indices [B, 1, H, W] (int64_t)
// Each thread block handles one (b, h, w) position, reduces over C.
__global__ void max_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ values,
                                int64_t* __restrict__ indices,
                                int B, int C, int H, int W,
                                long long sx_b, long long sx_c,
                                long long sx_h, long long sx_w,
                                long long so_b, long long so_c,
                                long long so_h, long long so_w)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    if (b >= B || h >= H || w >= W) return;

    const float* base = x + (long long)b * sx_b + (long long)h * sx_h + (long long)w * sx_w;
    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int c = 0; c < C; ++c) {
        float val = base[c * sx_c];
        if (val > max_val) {
            max_val = val;
            max_idx = c;
        }
    }

    long long out_idx = (long long)b * so_b + (long long)h * so_h + (long long)w * so_w;
    values[out_idx] = max_val;
    indices[out_idx] = (int64_t)max_idx;
}

// Kernel for max over dim=2, keepdim=False
// Input: x [B, 1, H, W] with strides (sx_b, sx_c, sx_h, sx_w)
// Output: values [B, 1, W], indices [B, 1, W] (int64_t)
__global__ void max_dim2_kernel(const float* __restrict__ x,
                                float* __restrict__ values,
                                int64_t* __restrict__ indices,
                                int B, int C, int H, int W,
                                long long sx_b, long long sx_c,
                                long long sx_h, long long sx_w,
                                long long so_b, long long so_c,
                                long long so_w)
{
    int b = blockIdx.x;
    int c = blockIdx.y; // should be 0, but we use C=1
    int w = blockIdx.z;
    if (b >= B || c >= C || w >= W) return;

    const float* base = x + (long long)b * sx_b + (long long)c * sx_c + (long long)w * sx_w;
    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int h = 0; h < H; ++h) {
        float val = base[h * sx_h];
        if (val > max_val) {
            max_val = val;
            max_idx = h;
        }
    }

    long long out_idx = (long long)b * so_b + (long long)c * so_c + (long long)w * so_w;
    values[out_idx] = max_val;
    indices[out_idx] = (int64_t)max_idx;
}

extern "C" void max_dim1_keepdim_launcher(
    const float* x, float* values, int64_t* indices,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    long long so_b, long long so_c, long long so_h, long long so_w,
    cudaStream_t stream)
{
    dim3 grid(B, H, W);
    max_dim1_kernel<<<grid, 1, 0, stream>>>(
        x, values, indices, B, C, H, W,
        sx_b, sx_c, sx_h, sx_w,
        so_b, so_c, so_h, so_w);
}

extern "C" void max_dim2_no_keepdim_launcher(
    const float* x, float* values, int64_t* indices,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    long long so_b, long long so_c, long long so_w,
    cudaStream_t stream)
{
    dim3 grid(B, C, W);
    max_dim2_kernel<<<grid, 1, 0, stream>>>(
        x, values, indices, B, C, H, W,
        sx_b, sx_c, sx_h, sx_w,
        so_b, so_c, so_w);
}