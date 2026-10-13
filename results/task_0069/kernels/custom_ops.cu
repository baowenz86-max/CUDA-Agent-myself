#include <cuda_runtime.h>
#include <math.h>

// Elementwise sigmoid: output[i] = 1 / (1 + exp(-input[i]))
__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Softmax over dim=1 (channels) for each batch and spatial position.
// Input tensor shape: [B, C, H, W] contiguous (row-major).
// One thread handles (b, h, w) position; loops over C channels.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int B, int C, int H, int W) {
    int HW = H * W;
    int total_spatial = B * HW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total_spatial; pos += stride) {
        int b = pos / HW;
        int hw = pos % HW; // index within H*W, corresponds to (h,w)
        int h = hw / W;
        int w = hw % W;

        const float* base = input + (int64_t)b * C * HW + hw;
        float* out_base = output + (int64_t)b * C * HW + hw;

        // Find max over channels for numerical stability
        float max_val = base[0];
        for (int c = 1; c < C; ++c) {
            float v = base[c * HW];
            if (v > max_val) max_val = v;
        }

        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += expf(base[c * HW] - max_val);
        }

        // Scale each channel
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < C; ++c) {
            out_base[c * HW] = expf(base[c * HW] - max_val) * inv_sum;
        }
    }
}

// Launcher for sigmoid: assumes contiguous float32 input, output allocated by caller
extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL; // grid dimension limit
    sigmoid_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw in extern "C" but we can record error and return; binding will check
        // We'll just ignore or perhaps add a simple flag? For simplicity, rely on binding check after sync.
    }
}

// Launcher for softmax over dim=1
extern "C" void softmax_dim1_forward_launcher(const float* input,
                                              float* output,
                                              int B, int C, int H, int W,
                                              cudaStream_t stream) {
    int HW = H * W;
    int64_t total_spatial = (int64_t)B * HW;
    if (total_spatial <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (total_spatial + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    softmax_dim1_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, B, C, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // same as above
    }
}
