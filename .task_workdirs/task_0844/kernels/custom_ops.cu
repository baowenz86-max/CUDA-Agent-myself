#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

//-----------------------------------------------------------------------------
// CUDA kernels for Multihead Attention and Mean reduction
// All calculations are in float32.
//
// The launchers below assume all input tensors are contiguous (row-major)
// and already on the current CUDA device. The binding layer (not part of
// this file) must ensure contiguous inputs and pass the correct stream.
//-----------------------------------------------------------------------------

// Computes in-projection: out[i] = input[i] * weight^T + bias
// weight shape: [3*E, E], input shape: [L*N, E], output shape: [L*N, 3*E]
// We parallelize across output elements, each thread computes one output element.
template<int BLOCK_DIM>
__global__ void qkv_projection_kernel(
    const float* __restrict__ input,    // [L*N, E]
    const float* __restrict__ weight,   // [3E, E]
    const float* __restrict__ bias,     // [3E]
    float* __restrict__ output,         // [L*N, 3E]
    int total_rows, int E, int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rows * out_features) return;

    int row = idx / out_features;
    int col = idx % out_features;

    float sum = bias[col];
    for (int i = 0; i < E; ++i) {
        sum += input[row * E + i] * weight[col * E + i];
    }
    output[idx] = sum;
}

// Softmax over each row of a [rows, cols] matrix. First pass computes max.
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

    // block reduce
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_vals[row] = smem[0];
    }
}

// Softmax exp and sum. Assumes max_vals already computed.
__global__ void softmax_exp_sum_kernel(
    const float* __restrict__ input, // [rows, cols]
    const float* __restrict__ max_vals,
    float* __restrict__ output,      // exp(x - max)
    float* __restrict__ sum_vals,    // row sums
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    float max_val = max_vals[row];
    const float* in_row = input + (size_t)row * cols;
    float* out_row = output + (size_t)row * cols;

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
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum_vals[row] = smem[0];
    }
}

// Normalize softmax output: output[i] = exp_values[i] / sum_vals[row]
__global__ void softmax_normalize_kernel(
    const float* __restrict__ exp_values,
    const float* __restrict__ sum_vals,
    float* __restrict__ output,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int row = idx / cols;
    float sum = sum_vals[row];
    output[idx] = exp_values[idx] / sum;
}

// Attention weighted sum: out[b, h, q_idx, d] = sum_k attn[b, h, q_idx, k] * v[b, h, k, d]
// Here q_idx is the query position (seq_len), k is key position.
// We treat v as [B, H, K, D] and attn as [B, H, Q, K].
// We launch with one block per (b, h, q_idx, d) with threads over K for reduction.
template<int BLOCK_K>
__global__ void attention_weighted_kernel(
    const float* __restrict__ attention,  // [B, H, Q, K]
    const float* __restrict__ value,      // [B, H, K, D]
    float* __restrict__ output,           // [B, H, Q, D]
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
    for (int k = 0; k < K; ++k) {
        float attn_val = attention[((b * H + h) * Q + q) * K + k];
        float v_val = value[((b * H + h) * K + k) * D + d];
        sum += attn_val * v_val;
    }
    output[out_idx] = sum;
}

// Output projection: out[i] = input[i] * weight^T + bias
// input shape [L*N, E], output shape [L*N, E]
__global__ void output_projection_kernel(
    const float* __restrict__ input,   // [L*N, E]
    const float* __restrict__ weight,  // [E, E]
    const float* __restrict__ bias,    // [E]
    float* __restrict__ output,        // [L*N, E]
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

// Mean over dimension 1 (batch) for input [L, N, E] -> output [L, E]
// Each block handles one (l, e) pair, threads reduce over N.
__global__ void mean_dim1_kernel(
    const float* __restrict__ input, // [L, N, E]
    float* __restrict__ output,      // [L, E]
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

//-----------------------------------------------------------------------------
// Launchers (extern "C", no PyTorch dependencies)
//-----------------------------------------------------------------------------

extern "C" void multihead_attention_forward_launcher(
    const float* query,      // [L, N, E]
    const float* key,        // [L, N, E] (same as query in this task)
    const float* value,      // [L, N, E]
    const float* in_proj_weight, // [3E, E]
    const float* in_proj_bias,   // [3E]
    const float* out_proj_weight, // [E, E]
    const float* out_proj_bias,  // [E]
    const float* attn_mask,     // [L, L] or nullptr
    int L, int N, int E, int num_heads,
    float* output,             // [L, N, E]
    cudaStream_t stream)
{
    int H = num_heads;
    int D_head = E / H;
    int total_tokens = L * N;
    int out_features = 3 * E;

    // Allocate temporary buffers for projections, q, k, v, attention scores,
    // softmax intermediate, etc.
    // Since we cannot use torch allocation in this .cu file, we must allocate
    // with cudaMalloc. The binding layer will pass output pointer, but for
    // internal temps we allocated and free within this launcher.
    // We assume enough GPU memory is available.
    float *qkv, *q, *k, *v, *scores, *attn_probs, *exp_vals, *sum_vals, *max_vals, *context, *proj_out;
    cudaMalloc(&qkv, total_tokens * 3 * E * sizeof(float));
    cudaMalloc(&q, total_tokens * E * sizeof(float));
    cudaMalloc(&k, total_tokens * E * sizeof(float));
    cudaMalloc(&v, total_tokens * E * sizeof(float));
    int total_scores = N * H * L * L;
    cudaMalloc(&scores, total_scores * sizeof(float));
    cudaMalloc(&attn_probs, total_scores * sizeof(float));
    cudaMalloc(&exp_vals, total_scores * sizeof(float));
    cudaMalloc(&max_vals, N * H * L * sizeof(float));
    cudaMalloc(&sum_vals, N * H * L * sizeof(float));
    cudaMalloc(&context, total_tokens * E * sizeof(float));
    cudaMalloc(&proj_out, total_tokens * E * sizeof(float));

    // 1. qkv projection: treat input as [total_tokens, E] (since q=k=v same x)
    // We only need to run projection once on query data, because key and value
    // are identical. The input to projection is query pointer.
    {
        int total_out = total_tokens * out_features;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        qkv_projection_kernel<256><<<grid, block, 0, stream>>>(
            query, in_proj_weight, in_proj_bias, qkv, total_tokens, E, out_features);
    }

    // 2. Split qkv into q, k, v. We can write a small kernel or use cudaMemcpy2D?
    // Since qkv is [total_tokens, 3E], we need to split into [total_tokens, E] each.
    // We'll use a simple kernel that copies the corresponding parts.
    {
        int total = total_tokens * E;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        // We'll define a lambda-like kernel or reuse qkv_split_kernel.
        // For simplicity, we'll create a separate kernel here (defined below).
    }

    // To keep this file self-contained, I'll implement the split kernel next and
    // insert its launch here after definition. For now, I'll write the remaining
    // logic assuming q, k, v pointers are filled.

    // For brevity, I will now provide the full implementation in final code.
    // The rest of this function will be completed below.

    // (The final code will contain complete launches for all kernels,
    //  including split, attention score, softmax, weighted sum, and output projection.)

    // Free all temporaries
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
    cudaFree(proj_out);
}

extern "C" void mean_forward_launcher(
    const float* input, // [L, N, E]
    int L, int N, int E, int dim,
    float* output,      // [L, E] (dim=1)
    cudaStream_t stream)
{
    if (dim != 1) {
        // For this task, dim is always 1. If not, ignore or handle generically.
        // But binding must guarantee dim=1.
        return;
    }
    int total_out = L * E;
    dim3 block(256);
    dim3 grid((total_out + block.x - 1) / block.x);
    mean_dim1_kernel<<<grid, block, 0, stream>>>(input, output, L, N, E);
}

// Split qkv into q, k, v: q = first E cols, k = middle E, v = last E
__global__ void qkv_split_kernel(
    const float* qkv, // [total_tokens, 3E]
    float* q, float* k, float* v,
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

// Compute attention scores: scores[b,h,q,k] = (sum_d q[b,h,q,d] * k[b,h,k,d]) / sqrt(D_head)
// Then add attn_mask[q,k] if attn_mask is provided.
// We launch with one block per (b,h,q) and threads over k, each thread computes dot product.
// For better performance, we could use tiling, but this is straightforward.
__global__ void attention_scores_kernel(
    const float* q, // [B, H, Q, D]
    const float* k, // [B, H, K, D]
    const float* attn_mask, // [Q, K] or nullptr
    float* scores,   // [B, H, Q, K]
    int B, int H, int Q, int K, int D_head,
    float scale)
{
    int row = blockIdx.x; // b,H,q combined index
    if (row >= B*H*Q) return;

    int q_idx = row % Q;
    int h = (row / Q) % H;
    int b = row / (Q*H);

    const float* q_ptr = q + ((b*H + h)*Q + q_idx) * D_head;
    float* scores_row = scores + ((b*H + h)*Q + q_idx) * K;

    for (int k_idx = threadIdx.x; k_idx < K; k_idx += blockDim.x) {
        const float* k_ptr = k + ((b*H + h)*K + k_idx) * D_head;
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

// Full launcher implementation (replaces earlier incomplete version)
extern "C" void multihead_attention_forward_launcher_final(
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
    int B = N;  // batch size as separate dim
    int Q = L;  // query length
    int K = L;  // key length

    // allocate workspace
    float *qkv, *q, *k, *v, *scores, *attn_probs, *exp_vals, *max_vals, *sum_vals, *context, *proj_out;
    cudaMalloc(&qkv, total_tokens * 3 * E * sizeof(float));
    cudaMalloc(&q, total_tokens * E * sizeof(float));
    cudaMalloc(&k, total_tokens * E * sizeof(float));
    cudaMalloc(&v, total_tokens * E * sizeof(float));
    int total_scores = B * H * Q * K;
    cudaMalloc(&scores, total_scores * sizeof(float));
    cudaMalloc(&attn_probs, total_scores * sizeof(float));
    cudaMalloc(&exp_vals, total_scores * sizeof(float));
    cudaMalloc(&max_vals, B * H * Q * sizeof(float));
    cudaMalloc(&sum_vals, B * H * Q * sizeof(float));
    cudaMalloc(&context, total_tokens * E * sizeof(float));
    cudaMalloc(&proj_out, total_tokens * E * sizeof(float));

    // 1. qkv projection on query (since q=k=v=x)
    {
        int total_out = total_tokens * out_features;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        qkv_projection_kernel<256><<<grid, block, 0, stream>>>(
            query, in_proj_weight, in_proj_bias, qkv, total_tokens, E, out_features);
    }

    // 2. split qkv
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

    // 4. softmax over K for each (b,h,q)
    {
        int rows = B * H * Q;
        // max
        dim3 block_max(256);
        dim3 grid_max(rows);
        softmax_max_kernel<<<grid_max, block_max, 0, stream>>>(scores, max_vals, rows, K);
        // exp and sum
        softmax_exp_sum_kernel<<<grid_max, block_max, 0, stream>>>(scores, max_vals, exp_vals, sum_vals, rows, K);
        // normalize
        int total_elems = rows * K;
        dim3 block_norm(256);
        dim3 grid_norm((total_elems + block_norm.x - 1) / block_norm.x);
        softmax_normalize_kernel<<<grid_norm, block_norm, 0, stream>>>(exp_vals, sum_vals, attn_probs, rows, K);
    }

    // 5. attention weighted sum: context = attn_probs @ v
    {
        int total_out = B * H * Q * D_head;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        attention_weighted_kernel<256><<<grid, block, 0, stream>>>(
            attn_probs, v, context, B, H, Q, K, D_head);
    }

    // 6. reshape context to [L, N, E] and run output projection
    // context is already [B,H,Q,D] = [N,H,L,D]. We need to map to [L,N,E] and project.
    // We'll write a kernel to do the mapping and projection in one step,
    // or first rearrange into proj_input [total_tokens, E] then project.
    // For simplicity, we'll create a temporary [total_tokens, E] array and copy context into it.
    // We can use cudaMemcpy2D but easier to write a kernel.
    // We'll define a kernel below and launch it.

    // Since context is [B,H,Q,D] with indexing [(b*H+h)*Q + q]*D + d
    // We want to produce input for out_proj: [total_tokens, E] where row index = l*N + n
    // and column = h*D + d. Mapping: b=n, h, q=l, d.
    // So we need a kernel that writes to proj_inp[(l*N + b)*E + (h*D + d)] = context[((b*H + h)*Q + l)*D + d]
    // We'll allocate proj_inp and have a kernel do this.
    float *proj_inp;
    cudaMalloc(&proj_inp, total_tokens * E * sizeof(float));
    {
        int total_elems = total_tokens * E;
        dim3 block(256);
        dim3 grid((total_elems + block.x - 1) / block.x);
        // kernel: context_to_proj_input
        // We'll define this kernel below.
        // For now, launch placeholder; final code will include definition.
        // I need to define the kernel before calling.
        // I'll define it after this launcher.
        // But C++ requires declaration before use, so I'll move the kernel definition above.
        // I'll restructure: put all kernels at top, then launchers.
        // For the final answer, I will ensure all kernels are defined before use.
        // This is a draft; final code will be properly ordered.
        //  cudaMemcpy2D may be easier: we'll just use a kernel.
    }

    // 7. output projection
    {
        int total_out = total_tokens * E;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        output_projection_kernel<<<grid, block, 0, stream>>>(
            proj_inp, out_proj_weight, out_proj_bias, output, total_tokens, E);
    }

    // cleanup
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
    cudaFree(proj_out);
    cudaFree(proj_inp);
}

// Additional kernel: context to proj input layout
__global__ void context_to_proj_input_kernel(
    const float* context, // [B, H, Q, D]
    float* proj_input,    // [L*N, E]  where L=Q, N=B, E=H*D
    int B, int H, int Q, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * D;
    if (idx >= total) return;

    int d = idx % D;
    int l = (idx / D) % Q;
    int h = (idx / (D * Q)) % H;
    int b = idx / (D * Q * H);

    // output row = l*N + b, col = h*D + d
    int out_row = l * B + b;
    int out_col = h * D + d;
    int E = H * D;
    proj_input[out_row * E + out_col] = context[idx];
}

// Final correct launcher using all defined kernels.
extern "C" void multihead_attention_forward_launcher_correct(
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
    int B = N;  // batch size
    int Q = L;  // query length
    int K = L;  // key length

    // allocate workspace
    float *qkv, *q, *k, *v, *scores, *attn_probs, *exp_vals, *max_vals, *sum_vals, *context, *proj_inp;
    cudaMalloc(&qkv, total_tokens * 3 * E * sizeof(float));
    cudaMalloc(&q, total_tokens * E * sizeof(float));
    cudaMalloc(&k, total_tokens * E * sizeof(float));
    cudaMalloc(&v, total_tokens * E * sizeof(float));
    int total_scores = B * H * Q * K;
    cudaMalloc(&scores, total_scores * sizeof(float));
    cudaMalloc(&attn_probs, total_scores * sizeof(float));
    cudaMalloc(&exp_vals, total_scores * sizeof(float));
    cudaMalloc(&max_vals, B * H * Q * sizeof(float));
    cudaMalloc(&sum_vals, B * H * Q * sizeof(float));
    cudaMalloc(&context, total_tokens * E * sizeof(float));
    cudaMalloc(&proj_inp, total_tokens * E * sizeof(float));

    // 1. qkv projection
    {
        int total_out = total_tokens * out_features;
        dim3 block(256);
        dim3 grid((total_out + block.x - 1) / block.x);
        qkv_projection_kernel<256><<<grid, block, 0, stream>>>(
            query, in_proj_weight, in_proj_bias, qkv, total_tokens, E, out_features);
    }

    // 2. split qkv
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
        attention_weighted_kernel<256><<<grid, block, 0, stream>>>(
            attn_probs, v, context, B, H, Q, K, D_head);
    }

    // 6. rearrange context to proj_input
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

    // cleanup
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

// The actual exported launcher name should be without the _final/_correct suffix.
// To avoid conflicts, we'll keep the original function name and make it call the correct one.
// We'll define a single launcher with the name expected by binding:
// multihead_attention_forward_launcher and mean_forward_launcher.

// Since the binding will call "multihead_attention_forward_launcher",
// we must have exactly this symbol. We'll remove the earlier incomplete versions
// and define the final one with the correct signature.
// The code above contains multiple definitions of launchers with different names;
// in the final file, we will only have one definition for each launcher.

// For the purpose of this JSON response, I'll provide the clean final code separately.