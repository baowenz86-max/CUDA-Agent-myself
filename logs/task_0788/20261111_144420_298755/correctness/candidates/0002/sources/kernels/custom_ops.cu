#include <cuda_runtime.h>
#include <math.h>

// Softmax over dimension 1 of a 4D tensor with shape (N, C, H, W).
// Each reduction slice consists of C elements, total slices = N*H*W.
// Kernel computes softmax independently for each slice.
// Input/output are contiguous float tensors.
__global__ void softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int total_slices = N * H * W;
    int slice_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (slice_idx >= total_slices) return;

    // Compute base index: slice_idx corresponds to (n, h, w) with linear ordering
    // The input is contiguous: index = n*C*H*W + c*H*W + h*W + w
    int n = slice_idx / (H * W);
    int rem = slice_idx % (H * W);
    int h = rem / W;
    int w = rem % W;
    int base = n * C * H * W + h * W + w;

    // Step 1: find max over channels
    float max_val = -INFINITY;
    for (int c = 0; c < C; ++c) {
        float val = input[base + c * H * W];
        if (val > max_val) max_val = val;
    }

    // Step 2: compute sum of exp(x - max)
    float sum_val = 0.0f;
    for (int c = 0; c < C; ++c) {
        float val = expf(input[base + c * H * W] - max_val);
        sum_val += val;
    }

    // Step 3: write output = exp(x - max) / sum
    float inv_sum = 1.0f / sum_val;
    for (int c = 0; c < C; ++c) {
        float val = expf(input[base + c * H * W] - max_val);
        output[base + c * H * W] = val * inv_sum;
    }
}

// Launcher function with C interface (no PyTorch dependencies).
// Parameters:
//   input: pointer to input tensor data (float32, contiguous)
//   output: pointer to output tensor data (same layout as input)
//   dim: reduction dimension (must be 1)
//   N, C, H, W: tensor dimensions
//   stream: CUDA stream for execution
extern "C" void custom_softmax_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream)
{
    // Only dim=1 supported
    if (dim != 1 || N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    int total_slices = (int)(N * H * W);
    const int block_size = 256;
    int grid_size = (total_slices + block_size - 1) / block_size;

    // Limit grid size to avoid overflow (max grid dim is 2^31-1, but total_slices is ~52M)
    // Launch kernel with grid-stride loop? Currently one thread per slice.
    // 52M slices, each thread processes C=64 elements; this is fine.
    // Grid size about 204800 blocks, within limits.
    softmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, (int)N, (int)C, (int)H, (int)W);
}
