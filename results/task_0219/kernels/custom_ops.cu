#include <cuda_runtime.h>
#include <cmath>

// Reduce last dimension (width) to mean, keepdim=True.
// Input: x with shape [B, C, H, W], arbitrary strides.
// Output: out with shape [B, C, H, 1], arbitrary strides.
// Kernel launch: one thread per output mean? Actually one block per output row could be inefficient for H large.
// Use one thread per output row, each thread sums along W sequentially. This is simple and works for any W.
// Grid size: total_rows = B*C*H. Block size 256. Grid-stride loop over rows.
// For W=1024, each thread does 1024 additions, okay.
__global__ void mean_dim_last_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    int B, int C, int H, int W,
    long long x_stride_b, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w
) {
    int total_rows = B * C * H;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int row = tid; row < total_rows; row += stride) {
        int b = row / (C * H);
        int rem = row - b * (C * H);
        int c = rem / H;
        int h = rem - c * H;
        const float* x_row = x + b * x_stride_b + c * x_stride_c + h * x_stride_h;
        float sum = 0.0f;
        for (int w = 0; w < W; ++w) {
            sum += x_row[w * x_stride_w];
        }
        float* out_ptr = out + b * out_stride_b + c * out_stride_c + h * out_stride_h;
        *out_ptr = sum / W;
    }
}

__global__ void softplus_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    long long numel,
    long long x_stride0, long long x_stride1, long long x_stride2, long long x_stride3,
    long long out_stride0, long long out_stride1, long long out_stride2, long long out_stride3,
    int ndim
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    // Softplus threshold as in PyTorch (20.0)
    const float threshold = 20.0f;
    for (; idx < numel; idx += stride) {
        // Compute linear indices for arbitrary ndim up to 4
        long long rem = idx;
        long long i3 = 0, i2 = 0, i1 = 0, i0 = 0;
        if (ndim == 4) {
            i3 = rem / x_stride3; rem %= x_stride3;
            i2 = rem / x_stride2; rem %= x_stride2;
            i1 = rem / x_stride1; rem %= x_stride1;
            i0 = rem;
        } else if (ndim == 3) {
            i2 = rem / x_stride2; rem %= x_stride2;
            i1 = rem / x_stride1; rem %= x_stride1;
            i0 = rem;
        } else if (ndim == 2) {
            i1 = rem / x_stride1; rem %= x_stride1;
            i0 = rem;
        } else { // ndim == 1
            i0 = rem;
        }
        const float* x_ptr = x + i3 * x_stride3 + i2 * x_stride2 + i1 * x_stride1 + i0 * x_stride0;
        float val = *x_ptr;
        float result = (val > threshold) ? val : log1pf(expf(val));
        float* out_ptr = out + i3 * out_stride3 + i2 * out_stride2 + i1 * out_stride1 + i0 * out_stride0;
        *out_ptr = result;
    }
}

extern "C" void mean_dim_last_launcher(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    long long x_stride_b, long long x_stride_c, long long x_stride_h, long long x_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream
) {
    int total_rows = B * C * H;
    if (total_rows <= 0 || W <= 0) return;
    int block = 256;
    int grid = (total_rows + block - 1) / block;
    // Cap grid to reasonable size; grid-stride loop handles rest
    if (grid > 65535) grid = 65535;
    mean_dim_last_kernel<<<grid, block, 0, stream>>>(
        out, x, B, C, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw from .cu without torch dependency. We could print, but better to let binding handle? We'll ignore here.
        // The binding will check stream sync for errors. We'll just do a printf for debugging.
        // printf("mean_dim_last kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void softplus_launcher(
    float* out,
    const float* x,
    long long numel,
    long long x_stride0, long long x_stride1, long long x_stride2, long long x_stride3,
    long long out_stride0, long long out_stride1, long long out_stride2, long long out_stride3,
    int ndim,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    int block = 256;
    long long grid_ll = (numel + block - 1) / block;
    int grid = (grid_ll > 65535) ? 65535 : (int)grid_ll;
    softplus_kernel<<<grid, block, 0, stream>>>(
        out, x, numel,
        x_stride0, x_stride1, x_stride2, x_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        ndim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore or debug print
    }
}
