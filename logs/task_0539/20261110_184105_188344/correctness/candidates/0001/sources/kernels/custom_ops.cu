#include <cuda_runtime.h>

// Kernel: sum over channel dimension (dim=1)
// Input: x [N, C, H, W] contiguous float32
// Output: out [N, H, W] float32
__global__ void sum_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        const float* base = x + n * C * H * W + h * W + w;
        float sum = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            sum += base[c * H * W];
        }
        out[i] = sum;
    }
}

// Kernel: elementwise ReLU
__global__ void relu_kernel(const float* __restrict__ x,
                            float* __restrict__ out,
                            int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float val = x[i];
        out[i] = val > 0.0f ? val : 0.0f;
    }
}

// Launcher for sum_dim1_forward
extern "C" void sum_dim1_launcher(const float* x,
                                  float* out,
                                  int N, int C, int H, int W,
                                  cudaStream_t stream) {
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid exceeding grid limits; grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(x, out, N, C, H, W);
}

// Launcher for relu_forward
extern "C" void relu_launcher(const float* x,
                               float* out,
                               int numel,
                               cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    relu_kernel<<<blocks, threads, 0, stream>>>(x, out, numel);
}
