#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for LPPool1d with norm_type=2 (Euclidean norm)
// input: 3D tensor [batch, channels, length]
// output: 3D tensor [batch, channels, output_length]
// kernel_size, stride are integers
template <typename scalar_t>
__global__ void lp_pool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int length,
    int output_length,
    int kernel_size,
    int stride) {

    // Each thread computes one output element
    int output_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int bc_idx = blockIdx.x * blockDim.x + threadIdx.x; // combined batch*channels index

    if (bc_idx >= batch * channels || output_idx >= output_length) {
        return;
    }

    int b = bc_idx / channels;
    int c = bc_idx % channels;

    int input_start = output_idx * stride;
    scalar_t sum_sq = 0.0f;

    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = input_start + k;
        if (input_pos < length) {
            scalar_t val = input[(b * channels + c) * length + input_pos];
            sum_sq += val * val;
        }
    }

    output[bc_idx * output_length + output_idx] = sqrtf(sum_sq);
}

// Launcher for lp_pool1d
extern "C" void lp_pool1d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int length,
    int output_length,
    int kernel_size,
    int stride,
    cudaStream_t stream) {

    // Grid dimensions: x for batch*channels, y for output_length
    dim3 threads_per_block(32, 8); // 256 threads per block
    dim3 num_blocks((batch * channels + threads_per_block.x - 1) / threads_per_block.x,
                    (output_length + threads_per_block.y - 1) / threads_per_block.y);

    lp_pool1d_kernel<float><<<num_blocks, threads_per_block, 0, stream>>>(
        input, output, batch, channels, length, output_length, kernel_size, stride);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Note: In a real binding layer, this would be reported via PyTorch error mechanisms.
        // Here we just ignore; the binding layer will check via torch.cuda.synchronize() later.
    }
}
