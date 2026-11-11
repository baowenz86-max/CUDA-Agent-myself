#include <cuda_runtime.h>

// Helper to compute ceiling division for grid size
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Kernel for max over dim=1 with keepdim=True.
// Input: x (float*) with shape [N, C, H, W] (contiguous)
// Output: out with shape [N, 1, H, W]
// Each thread computes one output element (n, 0, h, w) by iterating over c.
__global__ void max_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W
) {
    int total = N * H * W;  // number of output elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decompose i into (n, h, w)
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        int n = tmp / H;

        const float* base = x + ((long long)n * C * H * W) + ((long long)h * W) + w;
        float max_val = base[0];
        // Loop over C dimensions, using stride H*W to jump between channels
        for (int c = 1; c < C; ++c) {
            float val = base[(long long)c * H * W];
            if (val > max_val) {
                max_val = val;
            }
        }
        // Output index: n*1*H*W + 0*H*W + h*W + w = n*H*W + h*W + w = i
        out[i] = max_val;
    }
}

// Launcher for max_dim_1_keepdim. Assumes contiguous float32 input.
extern "C" void max_dim_1_keepdim_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    if (N == 0 || C == 0 || H == 0 || W == 0) {
        return;
    }
    int total = N * H * W;
    int threads = 256;
    int blocks = std::min(65535, ceil_div(total, threads));
    max_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(x, out, N, C, H, W);
}

// Kernel for triu on last two dimensions.
// Input: x (float*) with shape [batch, ..., rows, cols] (contiguous)
// Output: out same shape.
// We iterate over all elements, checking whether col >= row + diagonal in the last two dims.
__global__ void triu_last_two_dims_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long total_elements,
    int rows,
    int cols,
    int diagonal
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total_elements; i += stride) {
        // Determine position within last two dims
        long long pos = i % (long long)(rows * cols);
        int row = pos / cols;
        int col = pos - (long long)row * cols;  // remainder
        if (col >= row + diagonal) {
            out[i] = x[i];
        } else {
            out[i] = 0.0f;
        }
    }
}

// Launcher for triu_last_two_dims. Assumes contiguous float32 input.
extern "C" void triu_last_two_dims_launcher(
    const float* x,
    float* out,
    long long total_elements,
    int rows,
    int cols,
    int diagonal,
    cudaStream_t stream
) {
    if (total_elements == 0 || rows == 0 || cols == 0) {
        return;
    }
    int threads = 256;
    long long num_blocks_ll = ceil_div((long long)total_elements, (long long)threads);
    int blocks = (num_blocks_ll > 65535) ? 65535 : (int)num_blocks_ll;
    triu_last_two_dims_kernel<<<blocks, threads, 0, stream>>>(
        x, out, total_elements, rows, cols, diagonal
    );
}
