#include <cuda_runtime.h>

// Mean over dimension 0 for input shape [B, C, H, W] with B=16.
// Output shape [C, H, W]. Each thread computes one output element by accumulating over B.
__global__ void mean_dim0_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = idx / (H * W);

    float sum = 0.0f;
    // input is contiguous with shape [B,C,H,W]
    // offset = (((b * C + c) * H + h) * W + w)
    const float* base = input + ((long long)c * H + h) * W + w;
    for (int b = 0; b < B; ++b) {
        sum += base[(long long)b * C * H * W];
    }
    output[idx] = sum / (float)B;
}

// Hardswish elementwise: x * min(max(x+3,0),6)/6
__global__ void hardswish_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total; i += stride) {
        float x = input[i];
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[i] = x * relu6 * (1.0f / 6.0f);
    }
}

// Launcher for mean over dim 0
extern "C" void mean_dim0_launcher(const float* input,
                                   float* output,
                                   int B, int C, int H, int W,
                                   cudaStream_t stream) {
    int total = C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim0_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W);
}

// Launcher for hardswish
extern "C" void hardswish_launcher(const float* input,
                                    float* output,
                                    long long total,
                                    cudaStream_t stream) {
    int threads = 256;
    // Use grid stride, cap blocks to avoid overflow and adjust for large total
    int blocks = (int)min((long long)((total + threads - 1) / threads), (long long)65535);
    // If total is huge, grid-stride loop handles it; ensure blocks > 0 if total >0
    if (blocks == 0) blocks = 1;
    hardswish_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
