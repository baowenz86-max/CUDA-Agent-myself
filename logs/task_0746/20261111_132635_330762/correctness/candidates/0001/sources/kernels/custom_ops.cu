#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// custom_softmax: softmax over channel dimension (dim=1)
// Input: x with shape [N, C, H, W] (contiguous)
// Output: y with same shape, softmax over C for each (n, h, w)
// -----------------------------------------------------------------------------

__global__ void softmax_dim1_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int N, int C, int H, int W) {
    int total_spatial = H * W;
    int total_positions = N * total_spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total_positions; pos += stride) {
        int n = pos / total_spatial;
        int spatial = pos - n * total_spatial;
        int base = n * C * total_spatial + spatial;

        // find max
        float max_val = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            float v = x[base + c * total_spatial];
            if (v > max_val) max_val = v;
        }

        // compute sum of exp(x - max)
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += expf(x[base + c * total_spatial] - max_val);
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < C; ++c) {
            y[base + c * total_spatial] = expf(x[base + c * total_spatial] - max_val) * inv_sum;
        }
    }
}

// -----------------------------------------------------------------------------
// custom_rot90: rotation for k=1 and axes=(2,3)
// Input: x shape [N, C, H, W] (contiguous), output shape [N, C, W, H] (contiguous)
// Mapping: out[n,c,w,h] = in[n,c,h,W-1-w]
// -----------------------------------------------------------------------------

__global__ void rot90_k1_axes23_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int N, int C, int H, int W) {
    int total_out = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_out; i += stride) {
        // output linear index decomposition: i = n*(C*W*H) + c*(W*H) + w*H + h
        int n = i / (C * W * H);
        int rem_n = i - n * (C * W * H);
        int c = rem_n / (W * H);
        int rem_c = rem_n - c * (W * H);
        int w = rem_c / H;
        int h = rem_c - w * H;

        int in_idx = n * (C * H * W) + c * (H * W) + (h) * W + (W - 1 - w);
        y[i] = x[in_idx];
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C" for binding)
// -----------------------------------------------------------------------------

extern "C" void custom_softmax_launcher(const float* x,
                                        float* y,
                                        int64_t N,
                                        int64_t C,
                                        int64_t H,
                                        int64_t W,
                                        cudaStream_t stream) {
    if (N == 0 || C == 0 || H == 0 || W == 0) return;

    int total_positions = static_cast<int>(N * H * W);
    // choose block size and grid size adequate for large data
    int block_size = 256;
    int grid_size = (total_positions + block_size - 1) / block_size;
    // limit grid size to avoid excessive blocks, grid-stride loop handles rest
    if (grid_size > 65535) grid_size = 65535;

    softmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W)
    );
}

extern "C" void custom_rot90_launcher(const float* x,
                                      float* y,
                                      int64_t N,
                                      int64_t C,
                                      int64_t H,
                                      int64_t W,
                                      cudaStream_t stream) {
    if (N == 0 || C == 0 || H == 0 || W == 0) return;

    int total_out = static_cast<int>(N * C * H * W);
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    rot90_k1_axes23_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W)
    );
}

// Optional: generic launchers for other k values or axes, but not needed by contract.
// The binding layer will pass k=1, axes=(2,3) exclusively.
