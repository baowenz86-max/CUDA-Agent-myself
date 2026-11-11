#include <cuda_runtime.h>

// Kernel for mean along dimension 'dim' (only support dim=1 as per contract)
// x shape: [B, C, H, W] (C=3, B=256, H=512, W=256)
// out shape: [B, H, W]
__global__ void mean_dim_kernel(const float* __restrict__ x, float* __restrict__ out, int B, int C, int H, int W) {
    int total = B * H * W;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int w = idx % W;
        int h = (idx / W) % H;
        int b = idx / (W * H);
        float sum = 0.0f;
        const float* slice = x + (b * C * H * W) + h * W + w;
        for (int c = 0; c < C; ++c) {
            sum += slice[c * H * W];
        }
        out[idx] = sum / C;
    }
}

// Kernel for cosine similarity along dim=1
extern "C" void cosine_similarity_launcher(const float* x1, const float* x2, float* out, int B, int D1, int D2, cudaStream_t stream);

// Declare mean dim launcher as extern "C" for binding
extern "C" void mean_dim_launcher(const float* x, float* out, int B, int C, int H, int W, cudaStream_t stream) {
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    mean_dim_kernel<<<blocks, threads, 0, stream>>>(x, out, B, C, H, W);
}

__global__ void cosine_similarity_kernel(const float* __restrict__ x1, const float* __restrict__ x2, float* __restrict__ out, int B, int D1, int D2) {
    int total = B * D2;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int w = idx % D2;
        int b = idx / D2;
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (int d = 0; d < D1; ++d) {
            float a = x1[(b * D1 + d) * D2 + w];
            float b_val = x2[(b * D1 + d) * D2 + w];
            dot += a * b_val;
            norm1 += a * a;
            norm2 += b_val * b_val;
        }
        float denominator = sqrtf(norm1) * sqrtf(norm2);
        out[idx] = (denominator > 1e-8f) ? (dot / denominator) : 0.0f;
    }
}

extern "C" void cosine_similarity_launcher(const float* x1, const float* x2, float* out, int B, int D1, int D2, cudaStream_t stream) {
    int total = B * D2;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cosine_similarity_kernel<<<blocks, threads, 0, stream>>>(x1, x2, out, B, D1, D2);
}
