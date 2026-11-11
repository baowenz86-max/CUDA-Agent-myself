#include <cuda_runtime.h>

// 2x2 max pooling with stride 2, no padding
// Input: [N, C, H, W] float32, Output: [N, C, H/2, W/2]
__global__ void maxpool2d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * (H / 2) * (W / 2);
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int ow = i % (W / 2);
        int oh = (i / (W / 2)) % (H / 2);
        int c = (i / ((W / 2) * (H / 2))) % C;
        int n = i / ((W / 2) * (H / 2) * C);

        int in_h = oh * 2;
        int in_w = ow * 2;

        int base = ((n * C + c) * H + in_h) * W + in_w;
        float max_val = input[base];
        float v1 = input[base + 1];
        float v2 = input[base + W];
        float v3 = input[base + W + 1];

        max_val = fmaxf(max_val, v1);
        max_val = fmaxf(max_val, v2);
        max_val = fmaxf(max_val, v3);

        output[i] = max_val;
    }
}

// Leaky ReLU: output = x if x >= 0 else negative_slope * x
__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float negative_slope,
                                  int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        float val = input[i];
        output[i] = (val >= 0.0f) ? val : negative_slope * val;
    }
}

// Launcher for maxpool2d_forward
extern "C" void maxpool2d_launcher(const float* input,
                                    float* output,
                                    int N, int C, int H, int W,
                                    cudaStream_t stream) {
    int total = N * C * (H / 2) * (W / 2);
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
}

// Launcher for leaky_relu_forward
extern "C" void leaky_relu_launcher(const float* input,
                                     float* output,
                                     float negative_slope,
                                     int total,
                                     cudaStream_t stream) {
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(input, output, negative_slope, total);
}
