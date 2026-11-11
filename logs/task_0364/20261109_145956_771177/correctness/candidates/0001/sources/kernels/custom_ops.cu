#include <cuda_runtime.h>

// ReLU kernel: elementwise max(x, 0)
__global__ void relu_kernel(float* out, const float* in, long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float v = in[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

// Cumsum along dim=1 for contiguous [B, C, H, W] tensor.
// Each thread handles one (b, h, w) and serially accumulates over c.
// The tensor is contiguous: index = ((b * C + c) * H + h) * W + w.
__global__ void cumsum_dim1_kernel(float* out, const float* in,
                                   int B, int C, int H, int W) {
    long long hw = (long long)H * W;
    long long spatial = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial >= hw) return;

    long long h = spatial / W;
    long long w = spatial % W;

    for (int b = 0; b < B; ++b) {
        long long base = ((long long)b * C * H + h) * W + w;
        float acc = 0.0f;
        for (int c = 0; c < C; ++c) {
            long long offset = c * hw;
            acc += in[base + offset];
            out[base + offset] = acc;
        }
    }
}

// Launcher for ReLU
// All parameters are raw pointers and scalars, no PyTorch dependencies.
extern "C" void custom_relu_launcher(float* output, const float* input,
                                     long long numel, cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    if (blocks == 0) return;
    relu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

// Launcher for cumsum dim=1
// Input and output are contiguous tensors of shape [B, C, H, W]
extern "C" void custom_cumsum_dim1_launcher(float* output, const float* input,
                                            int B, int C, int H, int W,
                                            cudaStream_t stream) {
    long long hw = (long long)H * W;
    int threads = 256;
    long long blocks_ll = (hw + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    if (blocks == 0) return;
    cumsum_dim1_kernel<<<blocks, threads, 0, stream>>>(output, input, B, C, H, W);
}

// No torch/ATen/c10/pybind includes as required.
