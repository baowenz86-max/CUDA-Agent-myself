// MaxPool1d CUDA kernel and launcher
#include <cuda_runtime.h>
#include <cstdint>

// Kernel: each thread computes one output element
template <typename T>
__global__ void maxpool1d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N,          // batch size
    int64_t C,          // channels
    int64_t L_in,       // input length
    int64_t L_out,      // output length
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    int64_t total = N * C * L_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += grid_stride) {
        // Decode indices: i -> (n, c, l_out)
        int64_t l_out = i % L_out;
        int64_t tmp = i / L_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Compute input window start (including padding)
        int64_t start = l_out * stride - padding;
        int64_t end = start + kernel_size; // exclusive

        // Clamp to valid input range
        int64_t valid_start = start < 0 ? 0 : start;
        int64_t valid_end = end > L_in ? L_in : end;

        // If window is entirely in padding (should not happen with typical settings), output -inf? 
        // For MaxPool1d with padding, windows outside input are -inf in torch, but with k<=L+2p, at least one valid.
        T max_val = -1e30; // initialize to very negative
        bool has_valid = false;

        const T* input_base = input + (n * C + c) * L_in;
        for (int64_t k = valid_start; k < valid_end; ++k) {
            T val = input_base[k];
            if (!has_valid || val > max_val) {
                max_val = val;
                has_valid = true;
            }
        }

        output[i] = max_val;
    }
}

// Launcher with fixed float32 (since contract only uses float32)
extern "C" void maxpool1d_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t L_in,
    int64_t L_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream)
{
    int64_t total = N * C * L_out;
    int block_size = 256;
    int grid_size = (int)((total + block_size - 1) / block_size);
    // Cap grid size to avoid exceeding max grid dimension (usually 2^31-1), but total is ~268M, grid ~1M, well within limits.
    if (grid_size > 65535) {
        grid_size = 65535; // use grid-stride loop for anything larger
    }
    maxpool1d_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, L_in, L_out, kernel_size, stride, padding);
}
