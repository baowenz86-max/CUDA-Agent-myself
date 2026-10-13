#include <cuda_runtime.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// One-hot kernel: input int64 [N,H,W], output int64 [N,H,W,C]
// -----------------------------------------------------------------------------
__global__ void one_hot_kernel(const int64_t* __restrict__ input,
                               int64_t* __restrict__ output,
                               int N, int H, int W, int C) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H * W;
    if (idx >= total) return;
    int64_t value = input[idx];
    // output flattened index: (n*H*W + h*W + w) * C + c
    int64_t out_base = idx * C;
    for (int c = 0; c < C; ++c) {
        output[out_base + c] = (value == c) ? (int64_t)1 : (int64_t)0;
    }
}

// -----------------------------------------------------------------------------
// Cast kernel: int64 -> float32
// -----------------------------------------------------------------------------
__global__ void cast_int64_to_float_kernel(const int64_t* __restrict__ input,
                                          float* __restrict__ output,
                                          int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    output[idx] = (float)input[idx];
}

// -----------------------------------------------------------------------------
// Pad kernel: input float32 [N,H,W,C], output [N,H+2*pad, W+2*pad, C+2*pad]
// Pad last three dimensions symmetrically with given value.
// -----------------------------------------------------------------------------
__global__ void pad_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int N, int H, int W, int C,
                           int pad,
                           float pad_value) {
    int out_H = H + 2 * pad;
    int out_W = W + 2 * pad;
    int out_C = C + 2 * pad;

    int64_t out_idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t out_numel = (int64_t)N * out_H * out_W * out_C;
    if (out_idx >= out_numel) return;

    // Decompose linear index into (n, oh, ow, oc)
    int oc = (int)(out_idx % out_C);
    int64_t tmp = out_idx / out_C;
    int ow = (int)(tmp % out_W);
    tmp = tmp / out_W;
    int oh = (int)(tmp % out_H);
    int n = (int)(tmp / out_H);

    // Determine if this output position is in padding region
    bool in_pad = (oc < pad) || (oc >= C + pad) ||
                  (ow < pad) || (ow >= W + pad) ||
                  (oh < pad) || (oh >= H + pad);

    if (in_pad) {
        output[out_idx] = pad_value;
    } else {
        int c = oc - pad;
        int w = ow - pad;
        int h = oh - pad;
        int64_t in_idx = ((int64_t)n * H + h) * W + w;
        in_idx = in_idx * C + c;
        output[out_idx] = input[in_idx];
    }
}

// -----------------------------------------------------------------------------
// Softmax kernel along dimension 1 (H dimension)
// Input/output: float32 [N, H, W, C]
// Each (n, :, w, c) column is normalized.
// -----------------------------------------------------------------------------
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int N, int H, int W, int C) {
    // One block per (n, w, c), threads iterate over H dimension
    int64_t col_idx = blockIdx.x;  // column index = (n*W + w)*C + c
    int64_t total_cols = (int64_t)N * W * C;
    if (col_idx >= total_cols) return;

    int c = (int)(col_idx % C);
    int64_t tmp = col_idx / C;
    int w = (int)(tmp % W);
    int n = (int)(tmp / W);

    // Compute max over H for numerical stability
    float max_val = -1e30f;
    for (int h = 0; h < H; ++h) {
        int64_t idx = ((int64_t)n * H + h) * W + w;
        idx = idx * C + c;
        float v = input[idx];
        if (v > max_val) max_val = v;
    }

    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int h = 0; h < H; ++h) {
        int64_t idx = ((int64_t)n * H + h) * W + w;
        idx = idx * C + c;
        sum_exp += __expf(input[idx] - max_val);
    }
    float inv_sum = 1.0f / sum_exp;

    // Write normalized values
    for (int h = 0; h < H; ++h) {
        int64_t idx = ((int64_t)n * H + h) * W + w;
        idx = idx * C + c;
        output[idx] = __expf(input[idx] - max_val) * inv_sum;
    }
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C" for stable linkage)
// -----------------------------------------------------------------------------
extern "C" void one_hot_launcher(const int64_t* input,
                                 int64_t* output,
                                 int N, int H, int W, int C,
                                 cudaStream_t stream) {
    int64_t total = (int64_t)N * H * W;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    one_hot_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, N, H, W, C);
}

extern "C" void float_launcher(const int64_t* input,
                               float* output,
                               int64_t numel,
                               cudaStream_t stream) {
    int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    cast_int64_to_float_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void pad_launcher(const float* input,
                             float* output,
                             int N, int H, int W, int C,
                             int pad,
                             float pad_value,
                             cudaStream_t stream) {
    int out_H = H + 2 * pad;
    int out_W = W + 2 * pad;
    int out_C = C + 2 * pad;
    int64_t out_numel = (int64_t)N * out_H * out_W * out_C;
    int threads = 256;
    int64_t blocks = (out_numel + threads - 1) / threads;
    pad_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, N, H, W, C, pad, pad_value);
}

extern "C" void softmax_launcher(const float* input,
                                 float* output,
                                 int N, int H, int W, int C,
                                 cudaStream_t stream) {
    int64_t total_cols = (int64_t)N * W * C;
    int threads = 256;
    int64_t blocks = (total_cols + threads - 1) / threads;
    softmax_dim1_kernel<<<(int)blocks, threads, 0, stream>>>(input, output, N, H, W, C);
}
