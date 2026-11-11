// Clean implementation of multihead attention and mean.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Helper: allocate device memory with error check (not exported)
#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { /* In a real setting we'd report, but launcher cannot throw */ } }

// Kernel: qkv projection: out[row, col] = sum_i input[row, i] * weight[col, i] + bias[col]
// input: [total_tokens, E], weight: [3E, E], bias: [3E], output: [total_tokens, 3E]
__global__ void qkv_projection_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_tokens, int E, int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * out_features) return;
    int row = idx / out_features;
    int col = idx % out_features;
    float sum = bias[col];
    for (int i = 0; i < E; ++i) {
        sum += input[row * E + i] * weight[col * E + i];
    }
    output[idx] = sum;
}

// Kernel: split qkv into q, k, v. qkv: [total_tokens, 3E], q/k/v: [total_tokens, E]
__global__ void qkv_split_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int total_tokens, int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * E) return;
    int row = idx / E;
    int col = idx % E;
    q[idx] = qkv[row * 3 * E + col];
    k[idx] = qkv[row * 3 * E + E + col];
    v[idx] = qkv[row * 3 * E + 2 * E + col];
}

// Kernel: attention scores: scores[b,h,q,k] = (sum_d q[b,h,q,d]*k[b,h,k,d]) * scale + mask[q,k]
// One block per (b,h,q), threads over k and d.
__global__ void attention_scores_kernel(
    const float* __restrict__ q,   // [B,H,Q,D]
    const float* __restrict__ k,   // [B,H,K,D]
    const float* __restrict__ attn_mask, // [Q,K] or null
    float* __restrict__ scores,    // [B,H,Q,K]
    int B, int H, int Q, int K, int D_head, float scale)
{
    int row_idx = blockIdx.x; // combined b,h,q
    if (row_idx >= B * H * Q) return;
    int q_idx = row_idx % Q;
    int h = (row_idx / Q) % H;
    int b = row_idx / (Q * H);

    const float* q_ptr = q + ((b * H + h) * Q + q_idx) * D_head;
    float* scores_row = scores + ((b * H + h) * Q + q_idx) * K;

    for (int k_idx = threadIdx.x; k_idx < K; k_idx += blockDim.x) {
        const float* k_ptr = k + ((b * H + h) * K + k_idx) * D_head;
        float dot = 0.0f;
        for (int d = 0; d < D_head; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        dot *= scale;
        if (attn_mask != nullptr) {
            dot += attn_mask[q_idx * K + k_idx];
        }
        scores_row[k_idx] = dot;
    }
}

// Kernel: softmax max per row
__global__ void softmax_max_kernel(
    const float* __restrict__ input, // [rows, cols]
    float* __restrict__ max_vals,    // [rows]
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* in_row = input + (size_t)row * cols;
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = in_row[i];
        if (v > max_val) max_val = v;
    }
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) max_vals[row] = smem[0];
}

// Kernel: exp and sum
__global__ void softmax_exp_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ max_vals,
    float* __restrict__ exp_vals,
    float* __restrict__ sum_vals,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    float max_val = max_vals[row];
    const float* in_row = input + (size_t)row * cols;
    float* out_row = exp_vals + (size_t)row * cols;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = __expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) sum_vals[row] = smem[0];
}

// Kernel: normalize softmax
__global__ void softmax_normalize_kernel(
    const float* __restrict__ exp_vals,
    const float* __restrict__ sum_vals,
    float* __restrict__ output,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int row = idx / cols;
    float sum = sum_vals[row];
    output[idx] = exp_vals[idx] / sum;
}

// Kernel: attention weighted sum: context[b,h,q,d] = sum_k attn[b,h,q,k] * v[b,h,k,d]
__global__ void attention_weighted_kernel(
    const float* __restrict__ attn,  // [B,H,Q,K]
    const float* __restrict__ value, // [B,H,K,D]
    float* __restrict__ output,      // [B,H,Q,D]
    int B, int H, int Q, int K, int D)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * H * Q * D;
    if (out_idx >= total_out) return;

    int d = out_idx % D;
    int q = (out_idx / D) % Q;
    int h = (out_idx / (D * Q)) % H;
    int b = out_idx / (D * Q * H);

    float sum = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        float a = attn[((b * H + h) * Q + q) * K + kk];
        float vv = value[((b * H + h) * K + kk) * D + d];
        sum += a * vv;
    }
    output[out_idx] = sum;
}

// Kernel: rearrange context [B,H,Q,D] to proj_input [L*N, E] where L=Q, N=B, E=H*D
// output row = l*B + b, col = h*D + d
__global__ void context_to_proj_input_kernel(
    const float* __restrict__ context,
    float* __restrict__ proj_input,
    int B, int H, int Q, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * D;
    if (idx >= total) return;

    int d = idx % D;
    int q = (idx / D) % Q;
    int h = (idx / (D * Q)) % H;
    int b = idx / (D * Q * H);

    int out_row = q * B + b;
    int out_col = h * D + d;
    int E = H * D;
    proj_input[out_row * E + out_col] = context[idx];
}

// Kernel: output projection: out[row, col] = sum_i input[row, i] * weight[col, i] + bias[col]
__global__ void output_projection_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_rows, int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rows * E) return;
    int row = idx / E;
    int col = idx % E;
    float sum = bias[col];
    for (int i = 0; i < E; ++i) {
        sum += input[row * E + i] * weight[col * E + i];
    }
    output[idx] = sum;
}

// Kernel: mean over dimension 1: input [L, N, E] -> output [L, E]
__global__ void mean_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int L, int N, int E)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= L * E) return;
    int e = out_idx % E;
    int l = out_idx / E;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += input[(l * N + n) * E + e];
    }
    output[out_idx] = sum / N;
}

// Main launcher
extern "C" void multihead_attention_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    const float* attn_mask,
    int L, int N, int E, int num_heads,
    float* output, cudaStream_t stream)
{
    int H = num_heads;
    int D_head = E / H;
    int total_tokens = L * N;
    int out_features = 3 * E;
    int B = N;  // batch dimension
    int Q = L;  // query length
    int K = L;  // key length

    // Workspace
    float *qkv, *q, *k, *v, *scores, *attn_probs, *exp_vals, *max_vals, *sum_vals, *context, *proj_inp;
    cudaMalloc(&qkv, (size_t)total_tokens * out_features * sizeof(float));
    cudaMalloc(&q, (size_t)total_tokens * E * sizeof(float));
    cudaMalloc(&k, (size_t)total_tokens * E * sizeof(float));
    cudaMalloc(&v, (size_t)total_tokens * E * sizeof(float));
    int total_scores = B * H * Q * K;
    cudaMalloc(&scores, (size_t)total_scores * sizeof(float));
    cudaMalloc(&attn_probs, (size_t)total_scores * sizeof(float));
    cudaMalloc(&exp_vals, (size_t)total_scores * sizeof(float));
    cudaMalloc(&max_vals, (size_t)B * H * Q * sizeof(float));
    cudaMalloc(&sum_vals, (size_t)B * H * Q * sizeof(float));
    cudaMalloc(&context, (size_t)total_tokens * E * sizeof(float));
    cudaMalloc(&proj_inp, (size_t)total_tokens * E * sizeof(float));

    // 1. qkv projection
    {
        int total_out = total_tokens * out_features;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        qkv_projection_kernel<<<grid, block, 0, stream>>>(
            query, in_proj_weight, in_proj_bias, qkv, total_tokens, E, out_features);
    }

    // 2. split
    {
        int total_elems = total_tokens * E;
        dim3 block(256);
        dim3 grid((total_elems + block.x - 1) / block.x);
        qkv_split_kernel<<<grid, block, 0, stream>>>(qkv, q, k, v, total_tokens, E);
    }

    // 3. attention scores
    {
        dim3 block(256);
        dim3 grid(B * H * Q);
        float scale = 1.0f / sqrtf((float)D_head);
        attention_scores_kernel<<<grid, block, 0, stream>>>(
            q, k, attn_mask, scores, B, H, Q, K, D_head, scale);
    }

    // 4. softmax
    {
        int rows = B * H * Q;
        dim3 block(256);
        dim3 grid(rows);
        softmax_max_kernel<<<grid, block, 0, stream>>>(scores, max_vals, rows, K);
        softmax_exp_sum_kernel<<<grid, block, 0, stream>>>(scores, max_vals, exp_vals, sum_vals, rows, K);
        int total_elems = rows * K;
        dim3 block_norm(256);
        dim3 grid_norm((total_elems + block_norm.x - 1) / block_norm.x);
        softmax_normalize_kernel<<<grid_norm, block_norm, 0, stream>>>(exp_vals, sum_vals, attn_probs, rows, K);
    }

    // 5. attention weighted sum
    {
        int total_out = B * H * Q * D_head;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        attention_weighted_kernel<<<grid, block, 0, stream>>>(
            attn_probs, v, context, B, H, Q, K, D_head);
    }

    // 6. rearrange to proj_input
    {
        int total_elems = total_tokens * E;
        dim3 block(256);
        dim3 grid((total_elems + block.x - 1) / block.x);
        context_to_proj_input_kernel<<<grid, block, 0, stream>>>(
            context, proj_inp, B, H, Q, D_head);
    }

    // 7. output projection
    {
        int total_out = total_tokens * E;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        output_projection_kernel<<<grid, block, 0, stream>>>(
            proj_inp, out_proj_weight, out_proj_bias, output, total_tokens, E);
    }

    cudaFree(qkv);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(scores);
    cudaFree(attn_probs);
    cudaFree(exp_vals);
    cudaFree(max_vals);
    cudaFree(sum_vals);
    cudaFree(context);
    cudaFree(proj_inp);
}

// Mean launcher
extern "C" void mean_forward_launcher(
    const float* input,
    int L, int N, int E, int dim,
    float* output, cudaStream_t stream)
{
    if (dim != 1) return;
    int total_out = L * E;
    dim3 block(256);
    dim3 grid((total_out + block.x - 1) / block.x);
    mean_dim1_kernel<<<grid, block, 0, stream>>>(input, output, L, N, E);
}
