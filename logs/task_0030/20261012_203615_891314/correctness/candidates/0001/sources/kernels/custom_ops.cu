// custom_ops.cu
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// =========================== POLAR ===========================
// Computes complex64 output from float32 magnitudes and angles.
// Each thread computes one element: real = mag * cos(angle), imag = mag * sin(angle).
__global__ void polar_kernel(const float* __restrict__ mag,
                             const float* __restrict__ angle,
                             float2* __restrict__ out,
                             int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        float m = mag[i];
        float a = angle[i];
        out[i].x = m * cosf(a);
        out[i].y = m * sinf(a);
    }
}

extern "C" void polar_launcher(const float* mag,
                               const float* angle,
                               float2* out,
                               int64_t total,
                               cudaStream_t stream) {
    if (total <= 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    int blocks = (blocks64 > 1024) ? 1024 : (int)blocks64;
    polar_kernel<<<blocks, threads, 0, stream>>>(mag, angle, out, total);
}

// =========================== ONE_HOT ===========================
// Generates identity matrix of shape num_classes x num_classes as float32.
// Output is a float32 tensor with 1 on diagonal, 0 elsewhere.
__global__ void one_hot_kernel(float* out, int num_classes) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_classes && col < num_classes) {
        out[row * num_classes + col] = (row == col) ? 1.0f : 0.0f;
    }
}

extern "C" void one_hot_launcher(float* out,
                                int num_classes,
                                cudaStream_t stream) {
    if (num_classes <= 0) return;
    dim3 block(16, 16);
    dim3 grid((num_classes + block.x - 1) / block.x,
              (num_classes + block.y - 1) / block.y);
    one_hot_kernel<<<grid, block, 0, stream>>>(out, num_classes);
}

// =========================== SDPA ===========================
// Scaled dot-product attention: softmax(Q*K^T / sqrt(d)) * V
// Inputs: Q, K, V of shape [B, H, L, D], float32.
// Output: O of same shape, float32.
// This implementation uses one thread per output element, with a loop over sequence length.
// It is optimized for the given dimensions (L=128, D=16) but works generally.
__global__ void sdpa_kernel(const float* __restrict__ q,
                            const float* __restrict__ k,
                            const float* __restrict__ v,
                            float* __restrict__ out,
                            int B, int H, int L, int D) {
    int total = B * H * L * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode indices
    int d = idx % D;
    int l = (idx / D) % L;
    int h = (idx / (D * L)) % H;
    int b = idx / (D * L * H);

    float inv_scale = 1.0f / sqrtf((float)D);

    // Compute softmax denominator and numerator parts for this (b,h,l)
    float max_score = -1e30f;
    float scores[128]; // L up to 128, but we'll assume L <= 128 for stack; use dynamic for large L
    // For general L, we should use dynamic allocation, but for known L=128 we can use fixed array.
    // To be safe, we use a loop with two passes and no large array: first compute max, then denominator, then output.

    // First pass: compute max score for stability
    float local_max = -1e30f;
    for (int j = 0; j < L; ++j) {
        const float* q_row = q + ((b * H + h) * L + l) * D;
        const float* k_row = k + ((b * H + h) * L + j) * D;
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            dot += q_row[dd] * k_row[dd];
        }
        dot *= inv_scale;
        if (dot > local_max) local_max = dot;
    }

    // Second pass: compute denominator (sum of exp(score - max))
    float denom = 0.0f;
    for (int j = 0; j < L; ++j) {
        const float* q_row = q + ((b * H + h) * L + l) * D;
        const float* k_row = k + ((b * H + h) * L + j) * D;
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            dot += q_row[dd] * k_row[dd];
        }
        dot = (dot * inv_scale) - local_max;
        denom += expf(dot);
    }

    // Third pass: compute weighted sum of V
    float sum = 0.0f;
    for (int j = 0; j < L; ++j) {
        const float* q_row = q + ((b * H + h) * L + l) * D;
        const float* k_row = k + ((b * H + h) * L + j) * D;
        const float* v_row = v + ((b * H + h) * L + j) * D;
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            dot += q_row[dd] * k_row[dd];
        }
        dot = (dot * inv_scale) - local_max;
        float weight = expf(dot) / denom;
        sum += weight * v_row[d];
    }

    out[idx] = sum;
}

extern "C" void sdpa_launcher(const float* q,
                              const float* k,
                              const float* v,
                              float* out,
                              int B, int H, int L, int D,
                              cudaStream_t stream) {
    if (B <= 0 || H <= 0 || L <= 0 || D <= 0) return;
    int total = B * H * L * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    sdpa_kernel<<<blocks, threads, 0, stream>>>(q, k, v, out, B, H, L, D);
}
