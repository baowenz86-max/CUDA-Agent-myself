#include <cuda_runtime.h>
#include <math.h>

// Kernel for log_softmax along dim=1 (C=3)
// Input tensor shape: [N, C, H, W], strides: [C*H*W, H*W, W, 1]
// Each thread processes one spatial location (n, h, w) across all C channels.
// Grid-stride loop over total spatial locations = N * H * W.
__global__ void log_softmax_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N, int C, int H, int W) {
    int HW = H * W;
    long long total = (long long)N * HW; // 1024 * 50176 = 51,380,224
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        int n = i / HW;
        int spatial = i - (long long)n * HW;
        // Compute max over channels
        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)n * C + c) * HW + spatial];
            if (val > max_val) max_val = val;
        }
        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)n * C + c) * HW + spatial];
            sum += expf(val - max_val);
        }
        float log_sum = logf(sum);
        // Write log_softmax values
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)n * C + c) * HW + spatial];
            output[((long long)n * C + c) * HW + spatial] = val - max_val - log_sum;
        }
    }
}

// Kernel for hardshrink with threshold = 0.5
__global__ void hardshrink_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  long long numel,
                                  float lambd) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        if (fabsf(val) <= lambd) {
            output[i] = 0.0f;
        } else {
            output[i] = val;
        }
    }
}

// Fused kernel: log_softmax + hardshrink
__global__ void log_softmax_hardshrink_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int N, int C, int H, int W,
                                              float lambd) {
    int HW = H * W;
    long long total = (long long)N * HW;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        int n = i / HW;
        int spatial = i - (long long)n * HW;
        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)n * C + c) * HW + spatial];
            if (val > max_val) max_val = val;
        }
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)n * C + c) * HW + spatial];
            sum += expf(val - max_val);
        }
        float log_sum = logf(sum);
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)n * C + c) * HW + spatial];
            float ls = val - max_val - log_sum;
            if (fabsf(ls) <= lambd) {
                output[((long long)n * C + c) * HW + spatial] = 0.0f;
            } else {
                output[((long long)n * C + c) * HW + spatial] = ls;
            }
        }
    }
}

// Launcher functions
extern "C" void log_softmax_hardshrink_launcher(const float* input, float* output,
                                                int N, int C, int H, int W,
                                                cudaStream_t stream) {
    int HW = H * W;
    long long total = (long long)N * HW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    // Use fused kernel with lambd=0.5
    log_softmax_hardshrink_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, 0.5f);
}

extern "C" void log_softmax_only_launcher(const float* input, float* output,
                                          int N, int C, int H, int W,
                                          cudaStream_t stream) {
    int HW = H * W;
    long long total = (long long)N * HW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}

extern "C" void hardshrink_only_launcher(const float* input, float* output,
                                         long long numel,
                                         cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    hardshrink_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel, 0.5f);
}
