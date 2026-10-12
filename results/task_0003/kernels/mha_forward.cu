#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>

__global__ void transpose_add_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int B, int S, int H, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * D;
    if (idx >= total) return;
    int d = idx % D;
    int tmp = idx / D;
    int s = tmp % S;
    tmp /= S;
    int h = tmp % H;
    int b = tmp / H;
    int E = H * D;
    int row = b * S + s;
    int col = h * D + d;
    output[idx] = input[row * E + col] + bias[col];
}

__global__ void attention_forward_kernel(
    const float* q,
    const float* k,
    const float* v,
    const bool* mask,
    float* merged_out,
    int B, int S, int H, int D,
    float scale
) {
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_k = smem + S * D;
    float* s_v = smem + 2 * S * D;
    float* s_scores = smem + 3 * S * D;

    int bid = blockIdx.x;
    int b = bid / H;
    int h = bid % H;

    const float* q_base = q + bid * S * D;
    const float* k_base = k + bid * S * D;
    const float* v_base = v + bid * S * D;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int total_q = S * D;

    for (int i = tid; i < total_q; i += nthreads) {
        s_q[i] = q_base[i];
        s_k[i] = k_base[i];
        s_v[i] = v_base[i];
    }
    __syncthreads();

    int total_scores = S * S;
    for (int idx = tid; idx < total_scores; idx += nthreads) {
        int i = idx / S;
        int j = idx % S;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += s_q[i * D + d] * s_k[j * D + d];
        }
        dot *= scale;
        if (mask[b * S + j]) {
            dot = -INFINITY;
        }
        s_scores[idx] = dot;
    }
    __syncthreads();

    int warp_id = tid / 32;
    int lane = tid % 32;
    int num_warps = nthreads / 32;

    for (int row = warp_id; row < S; row += num_warps) {
        float* score_row = s_scores + row * S;
        float max_val = -INFINITY;
        for (int j = lane; j < S; j += 32) {
            float val = score_row[j];
            if (val > max_val) max_val = val;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        max_val = __shfl_sync(0xffffffff, max_val, 0);

        float sum = 0.0f;
        for (int j = lane; j < S; j += 32) {
            float val = score_row[j];
            if (val != -INFINITY) {
                sum += expf(val - max_val);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        sum = __shfl_sync(0xffffffff, sum, 0);

        for (int j = lane; j < S; j += 32) {
            float val = score_row[j];
            if (val != -INFINITY) {
                score_row[j] = expf(val - max_val) / sum;
            } else {
                score_row[j] = 0.0f;
            }
        }
    }
    __syncthreads();

    int E = H * D;
    float* out_base = merged_out + b * S * E;
    int total_out = S * D;
    for (int idx = tid; idx < total_out; idx += nthreads) {
        int i = idx / D;
        int d = idx % D;
        float acc = 0.0f;
        for (int j = 0; j < S; ++j) {
            acc += s_scores[i * S + j] * s_v[j * D + d];
        }
        out_base[i * E + h * D + d] = acc;
    }
}

__global__ void output_fused_kernel(
    const float* out_proj,
    const float* out_proj_bias,
    const float* x2,
    const float* x3,
    float* output,
    int out_d0, int out_d1, int out_d2, int out_d3,
    int B, int S, int E, int stack_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_d0 * out_d1 * out_d2 * out_d3;
    if (idx >= total) return;
    int tmp = idx;
    int d3 = tmp % out_d3; tmp /= out_d3;
    int d2 = tmp % out_d2; tmp /= out_d2;
    int d1 = tmp % out_d1; tmp /= out_d1;
    int d0 = tmp;

    bool from_x3;
    int batch, row, col;
    if (stack_dim == 0) {
        from_x3 = (d0 == 0);
        batch = d1; row = d2; col = d3;
    } else if (stack_dim == 1) {
        from_x3 = (d1 == 0);
        batch = d0; row = d2; col = d3;
    } else if (stack_dim == 2) {
        from_x3 = (d2 == 0);
        batch = d0; row = d1; col = d3;
    } else {
        from_x3 = (d3 == 0);
        batch = d0; row = d1; col = d2;
    }

    int src_idx = (batch * S + row) * E + col;
    if (from_x3) {
        output[idx] = x3[src_idx];
    } else {
        float val = out_proj[src_idx] + out_proj_bias[col];
        output[idx] = val * x2[src_idx];
    }
}

extern "C" void mha_forward_launcher(
    const float* x1, const float* x2, const float* x3,
    const bool* key_padding_mask,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* output,
    float* Q, float* K, float* V,
    float* q_t, float* k_t, float* v_t,
    float* merged_out,
    float* out_proj,
    int B, int S, int E, int H, int stack_dim,
    cudaStream_t stream
) {
    int D = E / H;
    int M = B * S;
    float alpha = 1.0f, beta = 0.0f;

    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    cublasSetStream(handle, stream);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, M, E,
                &alpha,
                in_proj_weight, E,
                x1, E,
                &beta,
                Q, E);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, M, E,
                &alpha,
                in_proj_weight + E * E, E,
                x2, E,
                &beta,
                K, E);

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, M, E,
                &alpha,
                in_proj_weight + 2 * E * E, E,
                x3, E,
                &beta,
                V, E);

    int total_t = B * H * S * D;
    int threads = 256;
    int blocks_t = (total_t + threads - 1) / threads;

    transpose_add_bias_kernel<<<blocks_t, threads, 0, stream>>>(
        Q, in_proj_bias, q_t, B, S, H, D);
    transpose_add_bias_kernel<<<blocks_t, threads, 0, stream>>>(
        K, in_proj_bias + E, k_t, B, S, H, D);
    transpose_add_bias_kernel<<<blocks_t, threads, 0, stream>>>(
        V, in_proj_bias + 2 * E, v_t, B, S, H, D);

    int attn_blocks = B * H;
    size_t smem_size = (3 * S * D + S * S) * sizeof(float);
    cudaFuncSetAttribute(attention_forward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    attention_forward_kernel<<<attn_blocks, threads, smem_size, stream>>>(
        q_t, k_t, v_t, key_padding_mask, merged_out,
        B, S, H, D, 1.0f / sqrtf((float)D));

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, M, E,
                &alpha,
                out_proj_weight, E,
                merged_out, E,
                &beta,
                out_proj, E);

    int out_d0, out_d1, out_d2, out_d3;
    if (stack_dim < 0) stack_dim += 4;
    if (stack_dim == 0) { out_d0=2; out_d1=B; out_d2=S; out_d3=E; }
    else if (stack_dim == 1) { out_d0=B; out_d1=2; out_d2=S; out_d3=E; }
    else if (stack_dim == 2) { out_d0=B; out_d1=S; out_d2=2; out_d3=E; }
    else if (stack_dim == 3) { out_d0=B; out_d1=S; out_d2=E; out_d3=2; }
    else { out_d0=B; out_d1=2; out_d2=S; out_d3=E; }

    int total_out = out_d0 * out_d1 * out_d2 * out_d3;
    int blocks_out = (total_out + threads - 1) / threads;
    output_fused_kernel<<<blocks_out, threads, 0, stream>>>(
        out_proj, out_proj_bias, x2, x3, output,
        out_d0, out_d1, out_d2, out_d3, B, S, E, stack_dim);
}
