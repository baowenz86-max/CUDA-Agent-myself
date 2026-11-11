#include <cuda_runtime.h>
#include <math.h>

// Fused linear + CELU kernel
// Each thread computes one output element (row-major, output[b][o])
//
// Parameters:
//   x: [batch_size, in_features] float32, supports arbitrary strides (but caller should pass contiguous pointer)
//   weight: [out_features, in_features] float32, row-major
//   bias: [out_features] float32
//   output: [batch_size, out_features] float32, row-major
//   in_features: number of input features, must be >0
//   out_features: number of output features
//   batch_size: batch dimension
//
// We assume x is contiguous (stride = [in_features, 1]) for performance;
// the binding layer must ensure contiguousness or provide a contiguous copy.
// In this task, x is from torch.randn which is contiguous.
__global__ void fused_linear_celu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each iteration handles one output feature index for the current batch row.
    // We use a 1D grid-stride loop over (batch * out_features) elements.
    // Alternative: 2D indexing, but 1D is simpler and coalesced for output writes.
    for (int idx = out_idx; idx < batch_size * out_features; idx += stride) {
        int b = idx / out_features;
        int o = idx - b * out_features;

        const float* x_row = x + (size_t)b * in_features;
        const float* w_row = weight + (size_t)o * in_features;

        // Compute dot product
        float sum = bias[o];
        // Use vectorized loads of float4 for better memory throughput
        int vec_limit = in_features / 4 * 4;
        for (int k = 0; k < vec_limit; k += 4) {
            float4 xv = *reinterpret_cast<const float4*>(x_row + k);
            float4 wv = *reinterpret_cast<const float4*>(w_row + k);
            sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
        }
        // Handle remaining elements
        for (int k = vec_limit; k < in_features; ++k) {
            sum += x_row[k] * w_row[k];
        }

        // Apply CELU with alpha = 1.0:
        // y = max(0, sum) + min(0, exp(sum) - 1)
        // Equivalent to: if sum > 0, y = sum; else y = expf(sum) - 1.0f;
        float y;
        if (sum > 0.0f) {
            y = sum;
        } else {
            y = expf(sum) - 1.0f;
        }

        output[idx] = y;
    }
}

// Launcher function (C interface, no PyTorch dependencies)
// Called from binding.cpp with raw pointers and stream.
extern "C" void linear_celu_forward(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
    // Total number of output elements
    long long total = (long long)batch_size * out_features;

    // Choose block size and grid size
    const int block = 256;
    // Use enough blocks to cover total elements, but cap grid to avoid exceeding max grid dim.
    // grid-stride loop handles any size.
    long long blocks_ll = (total + block - 1) / block;
    // Cap to avoid launching too many blocks, but grid-stride loop handles rest.
    // A reasonable cap is 65535 or something under max gridDim.x (2^31-1).
    // However for best occupancy, we can launch as many blocks as needed up to a few million.
    // Use min(blocks_ll, 65535) to avoid excessive block scheduling? Actually launch as many as needed if within limits.
    // Max gridDim.x for CUDA is 2^31-1, so for our sizes it's fine to launch all blocks.
    // But with total ~8M, blocks ~32768, well within limit. So we can launch all.
    int blocks = (int)blocks_ll;
    if (blocks < 1) blocks = 1;

    // Launch kernel
    fused_linear_celu_kernel<<<blocks, block, 0, stream>>>(
        x, weight, bias, output,
        batch_size, in_features, out_features);
}
