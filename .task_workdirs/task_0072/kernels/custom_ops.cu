#include <cuda_runtime.h>
#include <cmath>

// Kernel: mean along a specified dimension
// Input: x of shape [B, C, H, W] (contiguous float32)
// dim: reduction dimension (must be in [0,3])
// Output: y of shape [B, H, W] if dim=1, etc. General formula supported.
// For performance, specialized for dim=1 but general implementation handles any dim by computing strides.
// We use one thread per output element, with a loop over the reduction dimension.
// Output elements are indexed linearly; input index computed via unraveling.
// This kernel assumes contiguous input and output; validates dim on host.
__global__ void mean_dim_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t total_output_elements,
                                int dim,
                                int64_t B, int64_t C, int64_t H, int64_t W) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = gridDim.x * blockDim.x;

    // Precompute strides for 4D contiguous layout: [C*H*W, H*W, W, 1]
    int64_t stride0 = C * H * W;
    int64_t stride1 = H * W;
    int64_t stride2 = W;
    int64_t stride3 = 1;

    // Map linear idx to output coordinates (excluding dim)
    for (int64_t out_idx = idx; out_idx < total_output_elements; out_idx += grid_size) {
        // Compute output multidim indices (we need to know which dim is reduced)
        int64_t rem = out_idx;
        int64_t out_coords[4];
        // Number of dimensions after reduction = 3, but easier to compute input coords directly.
        // We'll compute input base index for each output element using generic method.
        // Since only dim=1 in practice, we optimize for that, but write generic for any dim.
        // Approach: reconstruct input indices without dim, then insert reduction index in loop.
        // Output linear index corresponds to linear index in a tensor of shape without dim.
        // We'll compute the input index base by expanding output linear index into coordinates.
        // Simpler: calculate all input coordinates for current output element, then sum over reduction dim.
        
        // Compute coordinates excluding dim.
        // For 4D, output shape is 3D, but we can handle by direct formulas:
        int64_t b,h,w;
        if (dim == 0) {
            // output shape [C,H,W]
            int64_t out_C = out_idx / (H * W);
            int64_t rem2 = out_idx % (H * W);
            h = rem2 / W;
            w = rem2 % W;
            b = 0; // will loop over b
            float sum = 0.0f;
            for (int64_t c = 0; c < B; ++c) {
                int64_t in_idx = c * stride0 + out_C * stride1 + h * stride2 + w;
                sum += input[in_idx];
            }
            output[out_idx] = sum / B;
        } else if (dim == 1) {
            // output shape [B,H,W]
            b = out_idx / (H * W);
            int64_t rem2 = out_idx % (H * W);
            h = rem2 / W;
            w = rem2 % W;
            float sum = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                int64_t in_idx = b * stride0 + c * stride1 + h * stride2 + w;
                sum += input[in_idx];
            }
            output[out_idx] = sum / C;
        } else if (dim == 2) {
            // output shape [B,C,W]
            b = out_idx / (C * W);
            int64_t rem2 = out_idx % (C * W);
            int64_t c = rem2 / W;
            w = rem2 % W;
            float sum = 0.0f;
            for (int64_t hh = 0; hh < H; ++hh) {
                int64_t in_idx = b * stride0 + c * stride1 + hh * stride2 + w;
                sum += input[in_idx];
            }
            output[out_idx] = sum / H;
        } else if (dim == 3) {
            // output shape [B,C,H]
            b = out_idx / (C * H);
            int64_t rem2 = out_idx % (C * H);
            int64_t c = rem2 / H;
            h = rem2 % H;
            float sum = 0.0f;
            for (int64_t ww = 0; ww < W; ++ww) {
                int64_t in_idx = b * stride0 + c * stride1 + h * stride2 + ww;
                sum += input[in_idx];
            }
            output[out_idx] = sum / W;
        }
        // Else invalid dim, but host validates.
    }
}

// Kernel: GELU elementwise (exact erf)
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    }
}

// Launcher for mean_dim
// input: pointer to float32 contiguous input
// output: pointer to allocated float32 output
// dim: reduction dimension as int64_t (host scalar, typically 1)
// B,C,H,W: input dimensions
// stream: cudaStream_t
extern "C" void mean_dim_launcher(const float* input, float* output, int64_t dim,
                                 int64_t B, int64_t C, int64_t H, int64_t W,
                                 cudaStream_t stream) {
    // Validate dim range
    if (dim < 0 || dim > 3) {
        // In case of invalid dim, we can do nothing or set error; binding should handle.
        // For safety, write zeros? No, we trust binding.
        return;
    }
    // Compute output size based on dim
    int64_t out_size;
    switch (dim) {
        case 0: out_size = C * H * W; break;
        case 1: out_size = B * H * W; break;
        case 2: out_size = B * C * W; break;
        case 3: out_size = B * C * H; break;
    }
    if (out_size == 0) return;

    const int block_size = 256;
    int64_t num_blocks = (out_size + block_size - 1) / block_size;
    // Cap grid to reasonable maximum (e.g., 4096 blocks) and use grid-stride loop
    if (num_blocks > 4096) num_blocks = 4096;

    mean_dim_kernel<<<(int)num_blocks, block_size, 0, stream>>>(
        input, output, out_size, (int)dim, B, C, H, W);
}

// Launcher for gelu
// input: pointer to float32 contiguous input
// output: pointer to allocated float32 output
// n: number of elements
// stream: cudaStream_t
extern "C" void gelu_launcher(const float* input, float* output, int64_t n,
                             cudaStream_t stream) {
    if (n == 0) return;
    const int block_size = 256;
    int64_t num_blocks = (n + block_size - 1) / block_size;
    if (num_blocks > 4096) num_blocks = 4096;
    gelu_kernel<<<(int)num_blocks, block_size, 0, stream>>>(input, output, n);
}
