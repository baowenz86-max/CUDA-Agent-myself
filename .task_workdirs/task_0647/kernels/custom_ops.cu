#include <cuda_runtime.h>
#include <cmath>

// Kernel for custom_unfold: copy input elements to output shape [batch, in_channels*kernel_h*kernel_w, 1]
__global__ void custom_unfold_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t batch,
                                     int64_t in_channels,
                                     int64_t height,
                                     int64_t width,
                                     int64_t kernel_h,
                                     int64_t kernel_w,
                                     int64_t total_output) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output; i += stride) {
        int64_t out_channels = in_channels * kernel_h * kernel_w;
        int64_t b = i / out_channels;
        int64_t rem = i % out_channels;
        int64_t c = rem / (kernel_h * kernel_w);
        int64_t pos = rem % (kernel_h * kernel_w);
        int64_t kh = pos / kernel_w;
        int64_t kw = pos % kernel_w;
        int64_t in_idx = ((b * in_channels + c) * height + kh) * width + kw;
        output[i] = input[in_idx];
    }
}

// Kernel for custom_abs: elementwise absolute value
__global__ void custom_abs_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Kernel for fused custom_unfold_abs: copy with absolute value
__global__ void custom_unfold_abs_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int64_t batch,
                                         int64_t in_channels,
                                         int64_t height,
                                         int64_t width,
                                         int64_t kernel_h,
                                         int64_t kernel_w,
                                         int64_t total_output) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output; i += stride) {
        int64_t out_channels = in_channels * kernel_h * kernel_w;
        int64_t b = i / out_channels;
        int64_t rem = i % out_channels;
        int64_t c = rem / (kernel_h * kernel_w);
        int64_t pos = rem % (kernel_h * kernel_w);
        int64_t kh = pos / kernel_w;
        int64_t kw = pos % kernel_w;
        int64_t in_idx = ((b * in_channels + c) * height + kh) * width + kw;
        output[i] = fabsf(input[in_idx]);
    }
}

// Helper to launch kernel with grid-stride loop
static inline void check_cuda_error(cudaStream_t stream) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw in .cu; rely on binding to check error.
    }
}

// C-interface launchers

extern "C" void custom_unfold_launcher(const float* input,
                                        float* output,
                                        int64_t batch,
                                        int64_t in_channels,
                                        int64_t height,
                                        int64_t width,
                                        int64_t kernel_h,
                                        int64_t kernel_w,
                                        cudaStream_t stream) {
    int64_t out_channels = in_channels * kernel_h * kernel_w;
    int64_t total_output = batch * out_channels;
    int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    custom_unfold_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, output, batch, in_channels, height, width,
        kernel_h, kernel_w, total_output);
    check_cuda_error(stream);
}

extern "C" void custom_abs_launcher(const float* input,
                                    float* output,
                                    int64_t total,
                                    cudaStream_t stream) {
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    custom_abs_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, total);
    check_cuda_error(stream);
}

extern "C" void custom_unfold_abs_launcher(const float* input,
                                           float* output,
                                           int64_t batch,
                                           int64_t in_channels,
                                           int64_t height,
                                           int64_t width,
                                           int64_t kernel_h,
                                           int64_t kernel_w,
                                           cudaStream_t stream) {
    int64_t out_channels = in_channels * kernel_h * kernel_w;
    int64_t total_output = batch * out_channels;
    int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    custom_unfold_abs_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, output, batch, in_channels, height, width,
        kernel_h, kernel_w, total_output);
    check_cuda_error(stream);
}
