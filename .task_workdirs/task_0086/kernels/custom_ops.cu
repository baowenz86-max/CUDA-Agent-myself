#include <cuda_runtime.h>
#include <cmath>

// -----------------------------------------------------------------------------
// ReflectionPad2d forward kernel
// Implements PyTorch nn.ReflectionPad2d exactly: output is reflection across
// input edges without repeating the edge pixel.
// Input: float32 CUDA tensor with arbitrary strides (supports non-contiguous).
// Output: contiguous tensor of shape [N, C, H+pad_top+pad_bottom, W+pad_left+pad_right].
// -----------------------------------------------------------------------------

__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W)
{
    long long total = (long long)N * C * out_H * out_W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        // decompose output index
        int ow = idx % out_W;
        long long tmp = idx / out_W;
        int oh = tmp % out_H;
        tmp /= out_H;
        int c = tmp % C;
        int n = tmp / C;

        int ih = oh - pad_top;
        int iw = ow - pad_left;

        // reflection mapping for height
        if (ih < 0) ih = -ih;  // since padding >=0, this reflects properly
        else if (ih >= H) ih = 2 * (H - 1) - ih;

        // reflection mapping for width
        if (iw < 0) iw = -iw;
        else if (iw >= W) iw = 2 * (W - 1) - iw;

        // The above works only if pad <= H-1 or W-1 respectively. If padding
        // is larger, multiple reflections are needed. For this specific task
        // pad=1 and H,W>=2, so single reflection is sufficient. To be robust
        // we could implement a while loop, but PyTorch requires pad <= H-1.

        long long in_idx = (long long)n * in_stride_N
                         + (long long)c * in_stride_C
                         + (long long)ih * in_stride_H
                         + (long long)iw * in_stride_W;
        output[idx] = input[in_idx];
    }
}

extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W,
    cudaStream_t stream)
{
    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;
    long long total = (long long)N * C * out_H * out_W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles rest
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W, out_H, out_W,
        pad_left, pad_right, pad_top, pad_bottom,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W);
}

// -----------------------------------------------------------------------------
// Scaled Dot Product Attention forward kernel
// Implements PyTorch F.scaled_dot_product_attention(q, k, v, is_causal)
// assuming q,k,v are 4D tensors of shape [N, H, L, D] and [N, H, S, D] and
// [N, H, S, V] respectively. For this task, L=S and V=D.
// Output shape equals q shape: [N, H, L, V].
// The scale factor is 1/sqrt(D).
// If is_causal is true, we apply causal mask -inf for positions j > i.
// The implementation uses one thread per output element (n, h, l, v).
// Naive implementation for simplicity; can be optimized later if needed.
// -----------------------------------------------------------------------------

__global__ void sdp_attention_kernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int N, int H, int L, int S, int D, int V,
    bool is_causal,
    float scale,
    long long q_stride_N, long long q_stride_H, long long q_stride_L, long long q_stride_D,
    long long k_stride_N, long long k_stride_H, long long k_stride_S, long long k_stride_D,
    long long v_stride_N, long long v_stride_H, long long v_stride_S, long long v_stride_V)
{
    long long total = (long long)N * H * L * V;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int v_idx = idx % V;
        long long tmp = idx / V;
        int l = tmp % L;
        tmp /= L;
        int h = tmp % H;
        int n = tmp / H;

        // base pointers for this batch and head
        const float* q_base = q + (long long)n * q_stride_N + (long long)h * q_stride_H;
        const float* k_base = k + (long long)n * k_stride_N + (long long)h * k_stride_H;
        const float* v_base = v + (long long)n * v_stride_N + (long long)h * v_stride_H;

        // compute scores for this query position l against all keys s
        // We allocate small arrays on stack to avoid repeated global reads.
        // D and S are likely small (D=32, S=1026). Max S maybe 1026 here.
        // Use fixed maximum capacity to avoid dynamic allocation in kernel.
        // For this task, S max is 1026, D=32, so we can use constants.
        // However, to be general, we use dynamic shared memory? For simplicity,
        // we recompute dot products and store scores in registers per thread? 
        // But each thread handles one output element, and needs entire row of scores.
        // Better approach: each thread block handles one or more query positions
        // cooperatively. Since this is a naive implementation, for correctness
        // we let each thread compute the whole row sequentially. This is slow
        // but correct. For performance, a tiled implementation is preferred.
        // Given the task is correctness-first, we implement a straightforward
        // sequential version per thread.

        float sum_exp = 0.0f;
        float acc = 0.0f;

        // Compute max score for numerical stability
        float max_score = -1e30f;
        for (int s = 0; s < S; ++s) {
            if (is_causal && s > l) {
                // masked position
                continue;
            }
            float dot = 0.0f;
            const float* q_ptr = q_base + (long long)l * q_stride_L;
            const float* k_ptr = k_base + (long long)s * k_stride_S;
            for (int d = 0; d < D; ++d) {
                dot += q_ptr[d * q_stride_D] * k_ptr[d * k_stride_D];
            }
            dot *= scale;
            if (dot > max_score) max_score = dot;
        }

        // Now compute exp and sum
        for (int s = 0; s < S; ++s) {
            if (is_causal && s > l) continue;
            float dot = 0.0f;
            const float* q_ptr = q_base + (long long)l * q_stride_L;
            const float* k_ptr = k_base + (long long)s * k_stride_S;
            for (int d = 0; d < D; ++d) {
                dot += q_ptr[d * q_stride_D] * k_ptr[d * k_stride_D];
            }
            dot = (dot * scale) - max_score;
            float exp_val = expf(dot);
            sum_exp += exp_val;
            // accumulate weighted value for this output dimension v_idx
            const float* v_ptr = v_base + (long long)s * v_stride_S + (long long)v_idx * v_stride_V;
            acc += exp_val * v_ptr[0];
        }

        output[idx] = acc / sum_exp;
    }
}

extern "C" void scaled_dot_product_attention_launcher(
    float* output,
    const float* q,
    const float* k,
    const float* v,
    int N, int H, int L, int S, int D, int V,
    bool is_causal,
    long long q_stride_N, long long q_stride_H, long long q_stride_L, long long q_stride_D,
    long long k_stride_N, long long k_stride_H, long long k_stride_S, long long k_stride_D,
    long long v_stride_N, long long v_stride_H, long long v_stride_S, long long v_stride_V,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)D);
    long long total = (long long)N * H * L * V;
    int threads = 128;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    sdp_attention_kernel<<<blocks, threads, 0, stream>>>(
        output, q, k, v,
        N, H, L, S, D, V,
        is_causal, scale,
        q_stride_N, q_stride_H, q_stride_L, q_stride_D,
        k_stride_N, k_stride_H, k_stride_S, k_stride_D,
        v_stride_N, v_stride_H, v_stride_S, v_stride_V);
}
