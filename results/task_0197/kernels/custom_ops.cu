#include <cuda_runtime.h>

// AvgPool2d kernel: kernel_size=2, stride=2, padding=0
// Input: [N, C, H, W] contiguous float32
// Output: [N, C, H_out, W_out] where H_out = H/2, W_out = W/2
__global__ void avgpool2d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int H, int W,
                                 int H_out, int W_out) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decode output linear index into n, c, h_out, w_out
        int w_out = i % W_out;
        int tmp = i / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Input indices for 2x2 pooling window
        int h_in = h_out * 2;
        int w_in = w_out * 2;

        // Base pointer for this (n,c) plane
        const float* in_plane = input + (n * C + c) * H * W;
        float sum = 0.0f;

        sum += in_plane[h_in * W + w_in];
        sum += in_plane[h_in * W + w_in + 1];
        sum += in_plane[(h_in + 1) * W + w_in];
        sum += in_plane[(h_in + 1) * W + w_in + 1];

        float avg = sum * 0.25f;
        output[i] = avg;
    }
}

// ReLU kernel: elementwise max(x, 0)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

// Host launcher for AvgPool2d
extern "C" void avgpool2d_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           int kernel_size, int stride, int padding,
                                           cudaStream_t stream) {
    // Only support kernel_size=2, stride=2, padding=0 for this model
    // Compute output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Grid-stride loop handles any size; cap blocks to a reasonable number
    if (blocks > 1024) blocks = 1024;

    avgpool2d_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, H_out, W_out);
}

// Host launcher for ReLU
extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int total,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
