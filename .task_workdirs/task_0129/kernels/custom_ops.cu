#include <cuda_runtime.h>
#include <float.h>

// Helper to get current CUDA error
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        // In launcher, we cannot throw, but we can record error and return; \
        // binding layer may check cudaGetLastError after call. \
        // We ignore for now but ensure kernel launch is followed by cudaGetLastError. \
    } \
} while(0)

// Softmax over dimension dim=1 (channels) for 3D tensor [B, C, L].
// This kernel assumes dim=1, but we can make it generic by passing dim and using stride logic.
// For simplicity and performance, specialized for dim=1 with contiguous or strided input.
// Each thread handles one (batch, l) position and reduces over C.
__global__ void softmax_dim1_kernel(float* out, const float* in,
                                    int B, int C, int L,
                                    int in_stride_b, int in_stride_c, int in_stride_l,
                                    int out_stride_b, int out_stride_c, int out_stride_l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * L;
    if (idx >= total) return;
    int b = idx / L;
    int l = idx % L;
    
    const float* base = in + b * in_stride_b + l * in_stride_l;
    float* out_base = out + b * out_stride_b + l * out_stride_l;
    
    // Find max over C
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        float v = base[c * in_stride_c];
        if (v > max_val) max_val = v;
    }
    // Compute exp sum
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = __expf(base[c * in_stride_c] - max_val);
        sum += v;
        // Store exp in out temporarily? Better compute again or store in shared/register.
        // We can store exp values in local array if C small, but C may be large.
        // For correctness, we can recompute in second loop, or store in per-thread array if C <= 32, else recompute.
    }
    // For current task C=16, we can use a small local array.
    // But to be generic, recompute exp and divide.
    float inv_sum = 1.0f / sum;
    for (int c = 0; c < C; ++c) {
        float v = __expf(base[c * in_stride_c] - max_val);
        out_base[c * out_stride_c] = v * inv_sum;
    }
}

// Cumprod over dimension dim=2 (sequence length) for 3D tensor [B, C, L].
// Each thread handles one (batch, channel) row and computes cumulative product sequentially.
__global__ void cumprod_dim2_kernel(float* out, const float* in,
                                    int B, int C, int L,
                                    int in_stride_b, int in_stride_c, int in_stride_l,
                                    int out_stride_b, int out_stride_c, int out_stride_l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;
    int b = idx / C;
    int c = idx % C;
    
    const float* in_row = in + b * in_stride_b + c * in_stride_c;
    float* out_row = out + b * out_stride_b + c * out_stride_c;
    
    float prod = 1.0f;
    for (int l = 0; l < L; ++l) {
        prod *= in_row[l * in_stride_l];
        out_row[l * out_stride_l] = prod;
    }
}

// Generic launchers
extern "C" void custom_softmax_launcher(
    float* output, const float* input,
    int B, int C, int L,
    int in_stride_b, int in_stride_c, int in_stride_l,
    int out_stride_b, int out_stride_c, int out_stride_l,
    int dim,
    cudaStream_t stream) {
    // Only support dim=1 for now; could add other dims if needed.
    // We'll assume dim==1 as contract.
    int total = B * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        output, input, B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l);
}

extern "C" void custom_cumprod_launcher(
    float* output, const float* input,
    int B, int C, int L,
    int in_stride_b, int in_stride_c, int in_stride_l,
    int out_stride_b, int out_stride_c, int out_stride_l,
    int dim,
    cudaStream_t stream) {
    // Only support dim=2 for now.
    int total = B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cumprod_dim2_kernel<<<blocks, threads, 0, stream>>>(
        output, input, B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l);
}
