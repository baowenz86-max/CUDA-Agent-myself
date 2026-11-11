// CUDA kernel for average pooling 2D with support for the task-specific configuration
#include <cuda_runtime.h>

// Kernel: average pooling over square kernel, stride, padding
// The kernel assumes input is contiguous and layout [N, C, H, W]
// output dimensions are computed outside in the launcher.
// This implementation is specialized for kernel_size=2, stride=2, padding=0 but general enough for other cases.
__global__ void avg_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t kernel_size, int64_t stride, int64_t padding,
                                  int64_t outH, int64_t outW) {
    // Compute output position (n, c, oh, ow) from block and thread indices
    // We use a 2D grid over output spatial dimensions and iterate over batch and channel via grid-stride,
    // or map batch/channel to blockIdx.z if available. For simplicity, use a 1D grid-stride loop over total output elements.
    int64_t total_output_elements = N * C * outH * outW;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_output_elements;
         idx += blockDim.x * gridDim.x) {
        // Decompose idx into n, c, oh, ow
        int64_t ow = idx % outW;
        int64_t tmp = idx / outW;
        int64_t oh = tmp % outH;
        tmp /= outH;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Input base offset for this channel and batch
        const float* in_base = input + (n * C + c) * H * W;

        // Compute pooling window start (inclusive) considering padding
        int64_t h_start = oh * stride - padding;
        int64_t w_start = ow * stride - padding;

        float sum = 0.0f;
        int count = 0;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            int64_t h_in = h_start + kh;
            bool h_valid = (h_in >= 0) && (h_in < H);
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t w_in = w_start + kw;
                bool w_valid = (w_in >= 0) && (w_in < W);
                if (h_valid && w_valid) {
                    sum += in_base[h_in * W + w_in];
                    ++count;
                }
            }
        }

        // Average: divisor is count (count_include_pad=False).
        // Since padding=0 and full windows cover output in this task, count == kernel_size*kernel_size.
        float result = (count > 0) ? (sum / count) : 0.0f;
        output[idx] = result;
    }
}

// C launcher function expected by binding.cpp
// Signature: void avg_pool2d_launcher(const float* input, float* output,
//                                     int64_t N, int64_t C, int64_t H, int64_t W,
//                                     int64_t kernel_size, int64_t stride, int64_t padding,
//                                     cudaStream_t stream)
extern "C" void float_avg_pool2d_launcher(const float* input, float* output,
                                          int64_t N, int64_t C, int64_t H, int64_t W,
                                          int64_t kernel_size, int64_t stride, int64_t padding,
                                          cudaStream_t stream) {
    // Compute output dimensions as per PyTorch AvgPool2d with ceil_mode=False
    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;

    // Total number of output elements
    int64_t total_elements = N * C * outH * outW;

    // Kernel launch parameters: 1D grid, 256 threads per block, grid-stride loop
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    // Cap blocks to avoid exceeding grid dimension limits (1D grid max 2^31-1)
    if (blocks > 65535) blocks = 65535;

    // Launch kernel
    avg_pool2d_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input, output, N, C, H, W, kernel_size, stride, padding, outH, outW);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will handle errors via TORCH_CHECK, but we can do a minimal check here.
        // Since we cannot include torch headers, we can't throw, but we can return with an error via a global flag or printf.
        // For correctness, the binding should check after calling the launcher.
        // We'll leave a simple printf for debugging, but not strictly needed.
        // printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }
}
