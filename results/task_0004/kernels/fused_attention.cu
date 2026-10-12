#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

__global__ void attention_context_kernel(
    const float* __restrict__ q_raw,
    const float* __restrict__ k_raw,
    const float* __restrict__ v_raw,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ context_out,
    int B, int T, int D, int H, int d_k, int repeat_interleave_times,
    float scale)
{
    int batch = blockIdx.x / H;
    int head = blockIdx.x % H;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];
    float* q_s = smem;
    float* k_s = smem + T * d_k;
    float* v_s = smem + 2 * T * d_k;

    const float* q_base = q_raw + batch * T * D + head * d_k;
    const float* k_base = k_raw + batch * T * D + head * d_k;
    const float* v_base = v_raw + batch * T * D + head * d_k;
    const float* bq = bias_q + head * d_k;
    const float* bk = bias_k + head * d_k;
    const float* bv = bias_v + head * d_k;

    int total = T * d_k;
    for (int idx = tid; idx < total; idx += block_size) {
        int t = idx / d_k;
        int d = idx % d_k;
        q_s[idx] = q_base[t * D + d] + bq[d];
    }
    for (int idx = tid; idx < total; idx += block_size) {
        int t = idx / d_k;
        int d = idx % d_k;
        k_s[idx] = k_base[t * D + d] + bk[d];
    }
    for (int idx = tid; idx < total; idx += block_size) {
        int t = idx / d_k;
        int d = idx % d_k;
        v_s[idx] = v_base[t * D + d] + bv[d];
    }
    __syncthreads();

    int L = T * repeat_interleave_times;
    int i = tid;
    if (i >= L) return;

    int orig_i = i / repeat_interleave_times;

    const int MAX_D_K = 64;
    float q[MAX_D_K];
    for (int d = 0; d < d_k; ++d) {
        q[d] = q_s[orig_i * d_k + d];
    }

    float max_score = -1.0e30f;
    for (int t = 0; t < T; ++t) {
        const float* k_t = k_s + t * d_k;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < d_k; ++d) {
            dot += q[d] * k_t[d];
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    float sum_exp = 0.0f;
    float context[MAX_D_K];
    #pragma unroll
    for (int d = 0; d < d_k; ++d) context[d] = 0.0f;

    for (int t = 0; t < T; ++t) {
        const float* k_t = k_s + t * d_k;
        const float* v_t = v_s + t * d_k;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < d_k; ++d) {
            dot += q[d] * k_t[d];
        }
        float score = dot * scale;
        float p = expf(score - max_score);
        float p_r = p * repeat_interleave_times;
        sum_exp += p_r;
        #pragma unroll
        for (int d = 0; d < d_k; ++d) {
            context[d] += p_r * v_t[d];
        }
    }

    float inv_sum = 1.0f / sum_exp;
    #pragma unroll
    for (int d = 0; d < d_k; ++d) context[d] *= inv_sum;

    float* out_row = context_out + batch * L * D + i * D + head * d_k;
    for (int d = 0; d < d_k; ++d) {
        out_row[d] = context[d];
    }
}

__global__ void add_bias_tan_kernel(float* output, const float* bias, int M, int D, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * D) {
        int col = idx % D;
        float val = output[idx] + bias[col];
        output[idx] = tanf(val * scaling_factor);
    }
}

extern "C" void fused_attention_launcher(
    const float* x, const float* y,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* q_raw, float* k_raw, float* v_raw,
    float* context, float* output,
    int B, int T, int D, int H, int repeat_interleave_times,
    float scaling_factor,
    cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int M = B * T;
    int K = D;
    int N = D;
    int L = T * repeat_interleave_times;

    const float* W_q = in_proj_weight;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                W_q, K,
                x, K,
                &beta,
                q_raw, N);

    const float* W_k = in_proj_weight + D * D;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                W_k, K,
                y, K,
                &beta,
                k_raw, N);

    const float* W_v = in_proj_weight + 2 * D * D;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                W_v, K,
                y, K,
                &beta,
                v_raw, N);

    int d_k = D / H;
    float scale = 1.0f / sqrtf((float)d_k);
    int blocks_attn = B * H;
    int threads_attn = 128;
    int shared_mem_size = T * d_k * 3 * sizeof(float);
    attention_context_kernel<<<blocks_attn, threads_attn, shared_mem_size, stream>>>(
        q_raw, k_raw, v_raw,
        in_proj_bias, in_proj_bias + D, in_proj_bias + 2 * D,
        context,
        B, T, D, H, d_k, repeat_interleave_times, scale
    );

    int M2 = B * L;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                D, M2, D,
                &alpha,
                out_proj_weight, D,
                context, D,
                &beta,
                output, D);

    int total_out = M2 * D;
    int threads_elem = 256;
    int blocks_elem = (total_out + threads_elem - 1) / threads_elem;
    add_bias_tan_kernel<<<blocks_elem, threads_elem, 0, stream>>>(
        output, out_proj_bias, M2, D, scaling_factor
    );

    cublasDestroy(handle);
}
