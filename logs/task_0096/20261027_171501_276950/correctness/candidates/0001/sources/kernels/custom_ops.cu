// CUDA kernels for custom operations.
// This file contains only non-convolutional CUDA kernels.
// ConvTranspose1d uses cuDNN and is implemented in the binding layer (kernels/conv_transpose1d_binding.cpp).
#include <cuda_runtime.h>

// Softmax over dim=1 for a 3D tensor of shape [batch, channels, length].
// Each thread block handles one (batch, length) position, computing softmax over channels.
// Uses shared memory reduction to compute max and sum, ensuring numerical stability.
// Assumes input is contiguous: stride[0] = channels*length, stride[1] = length, stride[2] = 1.
template <int BLOCK_SIZE>
__global__ void softmax_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int channels,
    int length)
{
    // Each block processes one (batch, length) position.
    // Block index: blockIdx.x = batch * length + l (if using 2D grid), else we use a 1D grid of size batch*length.
    // We launch grid of blocks equal to batch * length, so blockIdx.x encodes both b and l.
    int idx = blockIdx.x;
    int b = idx / length;
    int l = idx % length;

    // Base pointer for this (b, l) column.
    const float* in_col = input + (size_t)b * channels * length + l;
    float* out_col = output + (size_t)b * channels * length + l;

    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int stride = BLOCK_SIZE;

    // Step 1: find max value across channels.
    float max_val = -INFINITY;
    for (int c = tid; c < channels; c += stride) {
        float v = in_col[(size_t)c * length];
        if (v > max_val) max_val = v;
    }

    // Store per-thread max into shared memory and reduce.
    smem[tid] = max_val;
    __syncthreads();

    // Reduction for max.
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }
    float block_max = smem[0];
    __syncthreads();

    // Step 2: compute exp(x - max) and sum.
    float sum = 0.0f;
    for (int c = tid; c < channels; c += stride) {
        float v = in_col[(size_t)c * length];
        float e = expf(v - block_max);
        sum += e;
        // Store exp in output temporarily; later divide by sum.
        out_col[(size_t)c * length] = e;
    }

    // Reduce sum across threads.
    smem[tid] = sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float total_sum = smem[0];
    __syncthreads();

    // Step 3: divide exp values by sum to get softmax probabilities.
    for (int c = tid; c < channels; c += stride) {
        out_col[(size_t)c * length] /= total_sum;
    }
}

// Launcher for softmax_dim1_forward.
// Parameters:
//   output: pointer to output tensor (float, CUDA, contiguous)
//   input: pointer to input tensor (float, CUDA, contiguous)
//   batch, channels, length: dimensions (channels is the softmax dimension)
//   stream: CUDA stream for kernel launch
extern "C" void softmax_dim1_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int length,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int total_positions = batch * length;
    dim3 grid(total_positions);
    dim3 block(BLOCK_SIZE);

    // Ensure BLOCK_SIZE is at least enough to cover typical channel counts.
    // For channels <= BLOCK_SIZE, reduction works fine; for larger, grid-stride loop handles it.
    softmax_dim1_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        output, input, batch, channels, length);
}
