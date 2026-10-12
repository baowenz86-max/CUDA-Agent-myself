// custom_ops.cu
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// =========================== POLAR ===========================
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

// =========================== SDPA ===========================
__global__ void sdpa_kernel(const float* __restrict__ q,
                            const float* __restrict__ k,
                            const float* __restrict__ v,
                            float* __restrict__ out,
                            int B, int H, int L, int D) {
    int total = B * H * L * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % D;
    int l = (idx / D) % L;
    int h = (idx / (D * L)) % H;
    int b = idx / (D * L * H);

    float inv_scale = 1.0f / sqrtf((float)D);

    float max_score = -1e30f;
    for (int j = 0; j < L; ++j) {
        const float* q_row = q + ((b * H + h) * L + l) * D;
        const float* k_row = k + ((b * H + h) * L + j) * D;
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            dot += q_row[dd] * k_row[dd];
        }
        dot *= inv_scale;
        if (dot > max_score) max_score = dot;
    }

    float denom = 0.0f;
    for (int j = 0; j < L; ++j) {
        const float* q_row = q + ((b * H + h) * L + l) * D;
        const float* k_row = k + ((b * H + h) * L + j) * D;
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            dot += q_row[dd] * k_row[dd];
        }
        dot = (dot * inv_scale) - max_score;
        denom += expf(dot);
    }

    float sum = 0.0f;
    for (int j = 0; j < L; ++j) {
        const float* q_row = q + ((b * H + h) * L + l) * D;
        const float* k_row = k + ((b * H + h) * L + j) * D;
        const float* v_row = v + ((b * H + h) * L + j) * D;
        float dot = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            dot += q_row[dd] * k_row[dd];
        }
        dot = (dot * inv_scale) - max_score;
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
