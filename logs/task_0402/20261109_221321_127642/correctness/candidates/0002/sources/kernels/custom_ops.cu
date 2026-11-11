// CUDA kernel for mean reduction over dim=0 of a 4D tensor [N, C, H, W]
// Input: float32, output: float32, shape [C, H, W]
// Contract: torch.mean(x, dim=0)

#include <cuda_runtime.h>

// Kernel: one thread per (c, h, w) output element, loops over N dimension
__global__ void mean_dim0_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int H, int W) {
    // output size = C * H * W
    int total = C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decode i into (c, h, w)
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        int c = tmp / H;

        // Accumulate over N
        float sum = 0.0f;
        // index of input[n, c, h, w] = ((n * C + c) * H + h) * W + w
        // Precompute offset for n=0
        int base_offset = (c * H + h) * W + w;
        for (int n = 0; n < N; ++n) {
            sum += input[n * (C * H * W) + base_offset];
        }
        output[i] = sum / static_cast<float>(N);
    }
}

// Launcher: no torch dependencies, takes raw pointers and stream
extern "C" void mean_dim0_launcher(const float* input,
                                   float* output,
                                   int N, int C, int H, int W,
                                   cudaStream_t stream) {
    int total = C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Grid-stride loop handles large sizes
    mean_dim0_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
}
