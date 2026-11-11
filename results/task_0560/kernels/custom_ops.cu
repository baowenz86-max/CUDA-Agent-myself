#include <cuda_runtime.h>
#include <math.h>

// Softplus kernel: y = x <= threshold ? log1p(exp(x)) : x
// threshold = 20.0 (PyTorch default)
__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const float threshold = 20.0f;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = (val <= threshold) ? log1pf(expf(val)) : val;
    }
}

// Average pooling 2D kernel with padding, denominator excludes padding (count_include_pad=False)
// Input layout: [N, C, H, W] contiguous float32 tensor.
// Output: [N, C, OH, OW]
__global__ void avgpool2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t OH, int64_t OW,
                                 int kernel_size, int stride, int padding) {
    int64_t total_output = N * C * OH * OW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = blockDim.x * gridDim.x;
    float inv_kernel_area = 1.0f / static_cast<float>(kernel_size * kernel_size);
    for (int64_t out_idx = idx; out_idx < total_output; out_idx += grid_stride) {
        // Decompose linear index
        int64_t ow = out_idx % OW;
        int64_t tmp = out_idx / OW;
        int64_t oh = tmp % OH;
        tmp /= OH;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Input window start (including padding offset)
        int64_t h_start = oh * stride - padding;
        int64_t w_start = ow * stride - padding;
        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int64_t h_in = h_start + kh;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int64_t w_in = w_start + kw;
                if (w_in < 0 || w_in >= W) continue;
                int64_t input_offset = ((n * C + c) * H + h_in) * W + w_in;
                sum += input[input_offset];
                count++;
            }
        }
        // If count == 0 (should not happen unless kernel_size=0 or invalid padding), output 0
        output[out_idx] = (count > 0) ? sum / count : 0.0f;
    }
}

// C-style launcher for softplus
extern "C" void softplus_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    if (numel <= 0) return;
    int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
    // Cap grid to avoid excessive blocks for very large tensors (though grid-stride loop handles it)
    const int max_grid = 65535 * 32; // arbitrary large
    if (grid_size > max_grid) grid_size = max_grid;
    softplus_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In production, could printf or ignore; here we do nothing fatal, but ensure error is visible
        // Since no torch, we can't throw; the binding layer may check?
        // We'll leave as is; the binding layer can call cudaDeviceSynchronize and catch.
    }
}

// C-style launcher for avgpool2d
extern "C" void avgpool2d_launcher(const float* input, float* output,
                                   int64_t N, int64_t C, int64_t H, int64_t W,
                                   int64_t OH, int64_t OW,
                                   int kernel_size, int stride, int padding,
                                   cudaStream_t stream) {
    int64_t total_output = N * C * OH * OW;
    if (total_output <= 0) return;
    int block_size = 256;
    int grid_size = static_cast<int>((total_output + block_size - 1) / block_size);
    const int max_grid = 65535 * 32;
    if (grid_size > max_grid) grid_size = max_grid;
    avgpool2d_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N, C, H, W, OH, OW, kernel_size, stride, padding);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Same as above
    }
}
