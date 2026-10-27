#include <cuda_runtime.h>
#include <cstdint>

// Kernel: computes argmax along dim=1 for input shape [N, C, H, W]
// Each thread computes one output element.
// grid: enough blocks to cover total_output = N * H * W
// block: 256 threads
__global__ void argmax_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t total_output,
    int64_t spatial_size, // H*W
    int channels
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t out_idx = idx; out_idx < total_output; out_idx += stride) {
        // decompose out_idx into n and spatial offset
        int64_t n = out_idx / spatial_size;
        int64_t spatial = out_idx % spatial_size;

        // base pointer for this batch and spatial location
        const float* base = input + n * channels * spatial_size + spatial;

        int64_t max_channel = 0;
        float max_val = base[0];

        for (int c = 1; c < channels; ++c) {
            float val = base[c * spatial_size];
            if (val > max_val) {
                max_val = val;
                max_channel = c;
            }
        }

        output[out_idx] = max_channel;
    }
}

// Launcher: called from binding
// input: device pointer to float tensor, contiguous (caller should ensure contiguous)
// output: device pointer to int64 tensor, allocated by caller
// N, C, H, W: dimensions (C is the argmax dimension)
// stream: CUDA stream
extern "C" void argmax_dim1_launcher(
    const float* input,
    int64_t* output,
    int64_t N,
    int C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
) {
    int64_t spatial_size = H * W;
    int64_t total_output = N * spatial_size;
    
    if (total_output == 0) return;

    // Choose block size and grid size
    int block_size = 256;
    int grid_size = static_cast<int>((total_output + block_size - 1) / block_size);
    // Cap grid size to avoid too many blocks (max grid dim x is 2^31-1)
    // For total_output up to about 67M, grid_size ~ 262144, within limits.
    // Keep grid size reasonable: use min with 65535 * 2 to allow grid stride loop.
    // Actually we can use grid-stride loop with a fixed number of blocks for better occupancy.
    int max_grid = 65535 * 32; // ~2M
    if (grid_size > max_grid) {
        grid_size = max_grid;
    }

    argmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, total_output, spatial_size, C);
}
