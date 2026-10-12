#include <cuda_runtime.h>
#include <math.h>

// relu_sqrt kernel: y = sqrt(relu(x))
__global__ void relu_sqrt_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float v = input[i];
        v = fmaxf(v, 0.0f);
        output[i] = sqrtf(v);
    }
}

// silu kernel: y = x * sigmoid(x)
__global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// Attention weights kernel: computes softmax(QK^T/sqrt(d_k)) for each head and batch
__global__ void attention_weights_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ in_proj_weight,
    const float* __restrict__ in_proj_bias,
    float* __restrict__ attn_weights, // output shape: (batch, num_heads, seq_len, seq_len)
    int seq_len,
    int batch,
    int embed_dim,
    int num_heads
) {
    int head_dim = embed_dim / num_heads;
    int q_idx = blockIdx.x;       // query seq index
    int b = blockIdx.y * blockDim.y + threadIdx.y; // batch index (if using 2D grid)
    int head = blockIdx.z;        // head index

    // We'll use 2D grid: x = seq_len, y = num_heads*batch, z = 1
    // Actually, gridDim.x = seq_len, gridDim.y = batch*num_heads
    // blockDim.x = seq_len? No, we want one block per (q_idx, batch, head) maybe, with threads over key positions.
    // Simpler: one block per (query seq, batch, head), blockDim.x = seq_len (<= 640), each thread computes one attention weight.
    if (threadIdx.x >= seq_len) return;
    int k_idx = threadIdx.x;

    int q_row = (q_idx * batch + b) * embed_dim;
    int k_row = (k_idx * batch + b) * embed_dim;
    // Compute q and k projections for this head (full head_dim dot product)
    // We'll compute for each thread its own q projection? Better: each thread computes full dot product for its assigned key, but that requires query projection over all head_dim dimensions. We can compute query projection once per block using shared memory or have each thread compute it independently. Since blockDim.x = seq_len (640), each thread can compute full dot product with its key vector; that's head_dim * seq_len * (embed_dim) operations, which is heavy but correct.

    // To avoid recomputation, we can precompute Q projections in shared memory: for head, head_dim values; each thread computes one dimension of Q, then sync, then each thread uses all Q dims.
    // But that's complex. Alternatively, we can use a separate kernel that first computes Q and K projections into workspace, then performs batched matrix multiplication. That would be more efficient but more code.

    // Given the current constraints, we can compute Q and K projections on the fly per thread.
    // For each thread (k_idx), compute dot product between Q_head[q_idx] and K_head[k_idx].
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        // query projection value for (head, d) at q_idx
        float q_val = 0.0f;
        const float* q_vec = query + q_row;
        const float* wq = in_proj_weight + (head * head_dim + d) * embed_dim;
        for (int e = 0; e < embed_dim; ++e) {
            q_val += q_vec[e] * wq[e];
        }
        q_val += in_proj_bias[head * head_dim + d];

        // key projection value for (head, d) at k_idx
        float k_val = 0.0f;
        const float* k_vec = key + k_row;
        const float* wk = in_proj_weight + (embed_dim + head * head_dim + d) * embed_dim;
        for (int e = 0; e < embed_dim; ++e) {
            k_val += k_vec[e] * wk[e];
        }
        k_val += in_proj_bias[embed_dim + head * head_dim + d];

        score += q_val * k_val;
    }
    score *= 1.0f / sqrtf((float)head_dim);

    // Store raw score temporarily; we'll need softmax over all keys for this query.
    // We can do online softmax within block: each thread computes its own exp, block reduces to get max and sum, then normalize.
    __shared__ float s_max[1];
    __shared__ float s_sum[1];
    __shared__ float s_scores[1024]; // max seq_len, but adjust: use extern shared or dynamic.
    if (threadIdx.x == 0) {
        s_max[0] = -INFINITY;
        s_sum[0] = 0.0f;
    }
    __syncthreads();

    // First pass: find max score
    float local_max = score;
    // Reduce within block to global max
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = s_scores[threadIdx.x + offset]; // we need to store scores first
        }
    }
    // This is getting complicated; instead, we'll store scores in shared memory, then use block reduction for max and sum.
    // But blockDim.x = seq_len <= 640, we can use simpler approach: store scores in shared array, then each thread computes its own normalization by first finding max sequentially? But that requires all threads to have all scores.
    // Since seq_len is moderate (640), we can store scores in shared memory, sync, then each thread loops over all scores to find max and sum. That's O(seq_len) per thread, acceptable.

    // Store score in shared memory
    extern __shared__ float sh_scores[];
    sh_scores[threadIdx.x] = score;
    __syncthreads();

    // Find max and sum
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; ++i) {
        max_score = fmaxf(max_score, sh_scores[i]);
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        sum_exp += expf(sh_scores[i] - max_score);
    }
    float prob = expf(score - max_score) / sum_exp;

    // Write attention weight to global memory: shape (batch, num_heads, seq_len, seq_len)
    int weight_idx = ((b * num_heads + head) * seq_len + q_idx) * seq_len + k_idx;
    attn_weights[weight_idx] = prob;
}

// MHA forward kernel (original, without storing attention weights)
__global__ void attention_kernel(
    const float* __restrict__ query, const float* __restrict__ key, const float* __restrict__ value,
    const float* __restrict__ in_proj_weight, const float* __restrict__ in_proj_bias,
    float* __restrict__ workspace, int seq_len, int batch, int embed_dim, int num_heads) {
    int head_dim = embed_dim / num_heads;
    int q_idx = blockIdx.x; // 0..seq_len*batch-1
    int head = blockIdx.y;
    int s = q_idx / batch;
    int b = q_idx % batch;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int q_row = (s * batch + b) * embed_dim;
    float q_val = 0.0f;
    for (int e = 0; e < embed_dim; ++e) {
        q_val += query[q_row + e] * in_proj_weight[(head * head_dim + d) * embed_dim + e];
    }
    q_val += in_proj_bias[head * head_dim + d];

    float scaling = 1.0f / sqrtf((float)head_dim);
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float attn_accum = 0.0f;

    __shared__ float s_partial_sh[64];

    for (int s_k = 0; s_k < seq_len; ++s_k) {
        int k_row = (s_k * batch + b) * embed_dim;
        float k_val = 0.0f;
        for (int e = 0; e < embed_dim; ++e) {
            k_val += key[k_row + e] * in_proj_weight[(embed_dim + head * head_dim + d) * embed_dim + e];
        }
        k_val += in_proj_bias[embed_dim + head * head_dim + d];

        float partial_dot = q_val * k_val * scaling;
        s_partial_sh[d] = partial_dot;
        __syncthreads();
        for (int offset = head_dim / 2; offset > 0; offset >>= 1) {
            if (d < offset && d + offset < head_dim) {
                s_partial_sh[d] += s_partial_sh[d + offset];
            }
            __syncthreads();
        }
        float total_score = s_partial_sh[0];

        float new_max = fmaxf(max_score, total_score);
        if (new_max > max_score) {
            sum_exp = sum_exp * expf(max_score - new_max);
        }
        float exp_score = expf(total_score - new_max);
        sum_exp += exp_score;
        max_score = new_max;

        float v_val = 0.0f;
        for (int e = 0; e < embed_dim; ++e) {
            v_val += value[k_row + e] * in_proj_weight[(2 * embed_dim + head * head_dim + d) * embed_dim + e];
        }
        v_val += in_proj_bias[2 * embed_dim + head * head_dim + d];

        attn_accum += v_val * exp_score;
        __syncthreads();
    }

    attn_accum /= sum_exp;
    int workspace_idx = (s * batch + b) * embed_dim + head * head_dim + d;
    workspace[workspace_idx] = attn_accum;
}

__global__ void out_proj_kernel(
    const float* __restrict__ workspace, const float* __restrict__ out_proj_weight,
    const float* __restrict__ out_proj_bias, float* __restrict__ output,
    int seq_len, int batch, int embed_dim) {
    int q_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= embed_dim) return;
    int row = q_idx * embed_dim + tid;
    float sum = 0.0f;
    for (int j = 0; j < embed_dim; ++j) {
        sum += workspace[q_idx * embed_dim + j] * out_proj_weight[tid * embed_dim + j];
    }
    output[row] = sum + out_proj_bias[tid];
}

extern "C" void relu_sqrt_forward(
    const float* input, float* output, int numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    relu_sqrt_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void silu_forward(
    const float* input, float* output, int numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    silu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void mha_forward(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* output, float* attn_weights, int seq_len, int batch, int embed_dim, int num_heads,
    cudaStream_t stream) {
    int head_dim = embed_dim / num_heads;
    // Allocate workspace
    float* workspace = nullptr;
    size_t workspace_size = (size_t)seq_len * batch * embed_dim * sizeof(float);
    cudaMalloc(&workspace, workspace_size);
    if (!workspace) return;

    // Launch attention kernel to compute context (output before out_proj)
    dim3 grid_attn(seq_len * batch, num_heads);
    dim3 block_attn(head_dim);
    attention_kernel<<<grid_attn, block_attn, 0, stream>>>(
        query, key, value, in_proj_weight, in_proj_bias, workspace, seq_len, batch, embed_dim, num_heads);

    // Launch out_proj kernel
    dim3 block_out(embed_dim);
    dim3 grid_out(seq_len * batch);
    out_proj_kernel<<<grid_out, block_out, 0, stream>>>(
        workspace, out_proj_weight, out_proj_bias, output, seq_len, batch, embed_dim);

    // Launch attention weights kernel if attn_weights pointer is not null
    if (attn_weights) {
        // grid: x = seq_len, y = batch * num_heads
        dim3 grid_weights(seq_len, batch * num_heads);
        dim3 block_weights(seq_len); // threads = seq_len (<= 1024)
        // Dynamic shared memory size: seq_len * sizeof(float)
        size_t shmem_size = seq_len * sizeof(float);
        attention_weights_kernel<<<grid_weights, block_weights, shmem_size, stream>>>(
            query, key, in_proj_weight, in_proj_bias, attn_weights, seq_len, batch, embed_dim, num_heads);
    }

    cudaFree(workspace);
}
