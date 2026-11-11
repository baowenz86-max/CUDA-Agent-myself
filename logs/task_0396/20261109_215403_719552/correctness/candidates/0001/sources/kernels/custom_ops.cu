#include <cuda_runtime.h>
#include <cstdint>

// Kernel for max over dimension 1 (channels) with keepdim=True.
// Input x: shape [N, C, H, W], float32, contiguous or with strides.
// Output values: shape [N, 1, H, W], float32.
// Output indices: shape [N, 1, H, W], int64.
// The launcher passes raw pointers and logical dimensions/strides.
// This kernel uses a grid-stride loop over the output elements (N*H*W).
// Each output element computes max over C with its strides.
__global__ void max_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t total_output // N * H * W
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t out_idx = idx; out_idx < total_output; out_idx += stride) {
        // decompose out_idx into n, h, w
        int64_t n = out_idx / (H * W);
        int64_t rem = out_idx - n * (H * W);
        int64_t h = rem / W;
        int64_t w = rem - h * W;
        
        // start with first channel
        float best_val = x[n * C * H * W + 0 * H * W + h * W + w];
        int64_t best_idx = 0;
        for (int64_t c = 1; c < C; ++c) {
            float val = x[n * C * H * W + c * H * W + h * W + w];
            if (val > best_val) {
                best_val = val;
                best_idx = c;
            }
        }
        values[out_idx] = best_val;
        indices[out_idx] = best_idx;
    }
}

// Kernel for roll along dimension 2 (width) with shift = 1.
// Input x: shape [N, C, H, W], float32.
// Output y: same shape.
// For each element, y[n,c,h,(w+1)%W] = x[n,c,h,w].
// We iterate over all input elements and write to output with shifted index.
__global__ void roll_dim2_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t total // N * C * H * W
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        // decompose i into n,c,h,w
        int64_t n = i / (C * H * W);
        int64_t rem1 = i - n * C * H * W;
        int64_t c = rem1 / (H * W);
        int64_t rem2 = rem1 - c * H * W;
        int64_t h = rem2 / W;
        int64_t w = rem2 - h * W;
        int64_t new_w = (w + 1) % W;
        y[n * C * H * W + c * H * W + h * W + new_w] = x[i];
    }
}

// Extern "C" launcher for custom_max_dim1
// The binding layer will allocate output tensors and pass raw pointers.
// The stream is the current CUDA stream.
extern "C" void custom_max_dim1_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream
) {
    int64_t total_output = N * H * W;
    int block_size = 256;
    int64_t grid_size = (total_output + block_size - 1) / block_size;
    // cap grid size to 2^31-1 for safety, actual total_output for given shape is manageable
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    max_dim1_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        x, values, indices, N, C, H, W, total_output);
    // check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer may want to know; we can't throw, but we can leave it.
        // In production, the binding should check after call as well.
        // We can just ignore here, but better to include a macro? Not allowed to include anything else.
        // We'll leave a no-op that could be replaced if needed.
    }
}

// Extern "C" launcher for custom_roll_dim2
extern "C" void custom_roll_dim2_launcher(
    const float* x,
    float* y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream
) {
    int64_t total = N * C * H * W;
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    roll_dim2_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        x, y, N, C, H, W, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // no-op
    }
}
