// CUDA kernels for custom_mean_dim1 and custom_leaky_relu
// All kernels and launchers are in this single .cu file.
// No torch/ATen/c10/pybind dependencies.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Custom mean over dimension 1 with keepdim=True
// Input: x of shape [N, C, H, W] (float32)
// Output: out of shape [N, 1, H, W] where out[n,0,h,w] = sum_c x[n,c,h,w] / C
// The input tensor is assumed contiguous in memory with row-major layout:
// x[n][c][h][w] is at offset n*C*H*W + c*H*W + h*W + w.
// The output is also contiguous with shape [N,1,H,W] (but we write only those positions).
// -----------------------------------------------------------------------------

// Kernel: each thread computes one output element (n, h, w), summing over C.
// Grid is 1D over N*H*W, block size 256. Grid-stride loop handles any size.
__global__ void mean_dim1_kernel(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int N, int C, int H, int W,
                                 float inv_C) {
    int total = N * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total; pos += stride) {
        // Decode n, h, w from flat output index pos.
        int w = pos % W;
        int tmp = pos / W;
        int h = tmp % H;
        int n = tmp / H;

        // Base offset for this (n,h,w) in x: n*C*H*W + h*W + w
        // We need to sum over c = 0..C-1 at offsets base + c*H*W.
        const float* x_base = x + n * C * H * W + h * W + w;
        float sum = 0.0f;

        // Vectorized sum over C using float4 loads if possible.
        // Each vector load covers 4 consecutive c values (assuming C divisible by 4).
        // For general C, we do vectorized part then remainder.
        int c = 0;
        int vec_count = C / 4;
        if (vec_count > 0) {
            // Reinterpret as float4 pointer; must ensure alignment.
            const float4* x_vec = reinterpret_cast<const float4*>(x_base);
            // The stride between successive c is H*W, which may not be multiple of 4 floats,
            // so we cannot use contiguous float4 loads across c unless H*W is divisible by 4.
            // Instead, we do a loop with offset c*H*W and load a float4 there; this requires
            // the address to be 16-byte aligned. Since x_base may not be 16-byte aligned,
            // we fall back to scalar loads for safety. We'll use a scalar approach first.
        }

        // Scalar accumulation for simplicity and correctness.
        // This is memory-bound but straightforward. We can optimize later if needed.
        for (c = 0; c < C; ++c) {
            sum += x_base[c * H * W];
        }

        // Write result: out[n,0,h,w] = sum / C (multiply by inv_C for speed).
        out[pos] = sum * inv_C;
    }
}

// Launcher for mean_dim1
// Input: x_ptr has N*C*H*W elements (contiguous)
// Output: out_ptr has N*1*H*W elements (contiguous)
// Returns void, errors are checked via cudaGetLastError after launch.
extern "C" void custom_mean_dim1_launcher(const float* x_ptr,
                                          float* out_ptr,
                                          int64_t N,
                                          int64_t C,
                                          int64_t H,
                                          int64_t W,
                                          cudaStream_t stream) {
    if (x_ptr == nullptr || out_ptr == nullptr) return;
    int64_t total_outputs = N * H * W;
    if (total_outputs <= 0) return;

    // Cast dimensions to int for kernel (safe given max tensor size limits)
    int N_i = static_cast<int>(N);
    int C_i = static_cast<int>(C);
    int H_i = static_cast<int>(H);
    int W_i = static_cast<int>(W);

    float inv_C = 1.0f / static_cast<float>(C_i);

    // Launch configuration
    int block_size = 256;
    int grid_size = static_cast<int>((total_outputs + block_size - 1) / block_size);
    // Cap grid size to avoid excessive blocks, grid-stride loop will handle.
    int max_grid_size = 65535; // common limit; could be larger on newer GPUs, but safe for now
    if (grid_size > max_grid_size) grid_size = max_grid_size;

    mean_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        x_ptr, out_ptr, N_i, C_i, H_i, W_i, inv_C);

    // The binding will check cudaGetLastError after launch.
}

// -----------------------------------------------------------------------------
// Leaky ReLU elementwise kernel
// Input: x of shape [N, 1, H, W] => total numel = N*H*W
// Output: same shape, each element y = x if x >= 0 else negative_slope * x
// -----------------------------------------------------------------------------
__global__ void leaky_relu_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int64_t numel,
                                  float negative_slope) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = x[i];
        y[i] = (val >= 0.0f) ? val : (negative_slope * val);
    }
}

// Launcher for leaky_relu
extern "C" void custom_leaky_relu_launcher(const float* x_ptr,
                                            float* y_ptr,
                                            int64_t numel,
                                            double negative_slope,
                                            cudaStream_t stream) {
    if (x_ptr == nullptr || y_ptr == nullptr || numel <= 0) return;

    int block_size = 256;
    int64_t grid_size_64 = (numel + block_size - 1) / block_size;
    int grid_size = static_cast<int>(grid_size_64);
    int max_grid_size = 65535;
    if (grid_size > max_grid_size) grid_size = max_grid_size;

    float slope_f = static_cast<float>(negative_slope);

    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        x_ptr, y_ptr, numel, slope_f);

    // cudaGetLastError checked externally.
}
