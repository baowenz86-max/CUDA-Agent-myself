#include <cuda_runtime.h>
#include <math.h>

__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Split projection kernel: computes q,k,v from query,key,value and writes to separate arrays.
__global__ void in_projection_split_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ in_proj_weight,
    const float* __restrict__ in_proj_bias,
    float* __restrict__ d_q,
    float* __restrict__ d_k,
    float* __restrict__ d_v,
    int seq_len, int batch, int embed_dim, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int head_dim = embed_dim / num_heads;
    int total = seq_len * batch * 3 * embed_dim;
    for (int i = idx; i < total; i += stride) {
        int e = i % embed_dim;
        int tmp = i / embed_dim;
        int proj_idx = tmp % 3;  // 0:q, 1:k, 2:v
        int tmp2 = tmp / 3;
        int b = tmp2 % batch;
        int s = tmp2 / batch;
        const float* input = (proj_idx == 0) ? query : (proj_idx == 1) ? key : value;
        // input is (seq_len, batch, embed_dim) => linear index (s*batch + b)*embed_dim + d
        const float* weight_row = in_proj_weight + (proj_idx * embed_dim + e) * embed_dim;
        float sum = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
            sum += input[(s * batch + b) * embed_dim + d] * weight_row[d];
        }
        if (in_proj_bias != nullptr) {
            sum += in_proj_bias[proj_idx * embed_dim + e];
        }
        // Write to q/k/v in shape (seq_len, batch*num_heads, head_dim)
        int h = e / head_dim;
        int d = e % head_dim;
        int out_idx = (s * (batch * num_heads) + (b * num_heads + h)) * head_dim + d;
        float* dest = (proj_idx == 0) ? d_q : (proj_idx == 1) ? d_k : d_v;
        dest[out_idx] = sum;
    }
}

// Attention output kernel: online softmax with tiling, one block per (batch*head).
__global__ void attention_output_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ attn_output,
    int seq_len, int batch, int num_heads, int head_dim,
    float scale) {
    extern __shared__ float smem[];
    int bh = blockIdx.x; // batch*heads index
    int tid = threadIdx.x;
    const int TILE_Q = blockDim.x;
    const int TILE_K = blockDim.x;

    float* K_tile = smem; // TILE_K * head_dim
    float* V_tile = K_tile + TILE_K * head_dim;
    float* q_tile = V_tile + TILE_K * head_dim;
    // Note: No need for acc in shared; we use registers per thread.

    int num_q_tiles = (seq_len + TILE_Q - 1) / TILE_Q;
    for (int q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
        int q_start = q_tile * TILE_Q;
        int q_idx = q_start + tid;
        bool valid_q = q_idx < seq_len;

        if (valid_q) {
            for (int d = 0; d < head_dim; ++d) {
                q_tile[tid * head_dim + d] = q[(q_idx * (batch * num_heads) + bh) * head_dim + d];
            }
        }
        __syncthreads();

        float max_val = -INFINITY;
        float sum_exp = 0.0f;
        float acc_reg[32]; // maximum head_dim assumed 32, but can be up to 64; use 64 to be safe
        for (int d = 0; d < head_dim; ++d) acc_reg[d] = 0.0f;

        int num_kv_tiles = (seq_len + TILE_K - 1) / TILE_K;
        for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
            int kv_start = kv_tile * TILE_K;
            int kv_idx = kv_start + tid;
            bool valid_kv = kv_idx < seq_len;

            if (valid_kv) {
                for (int d = 0; d < head_dim; ++d) {
                    K_tile[tid * head_dim + d] = k[(kv_idx * (batch * num_heads) + bh) * head_dim + d];
                    V_tile[tid * head_dim + d] = v[(kv_idx * (batch * num_heads) + bh) * head_dim + d];
                }
            } else {
                for (int d = 0; d < head_dim; ++d) {
                    K_tile[tid * head_dim + d] = 0.0f;
                    V_tile[tid * head_dim + d] = 0.0f;
                }
            }
            __syncthreads();

            if (valid_q) {
                for (int kk = 0; kk < TILE_K; ++kk) {
                    if (kv_start + kk >= seq_len) break;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q_tile[tid * head_dim + d] * K_tile[kk * head_dim + d];
                    }
                    score *= scale;
                    if (score > max_val) {
                        float correction = expf(max_val - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        for (int d = 0; d < head_dim; ++d) acc_reg[d] *= correction;
                        max_val = score;
                    } else {
                        float weight = expf(score - max_val);
                        sum_exp += weight;
                    }
                    float weight = expf(score - max_val);
                    for (int d = 0; d < head_dim; ++d) {
                        acc_reg[d] += weight * V_tile[kk * head_dim + d];
                    }
                }
            }
            __syncthreads();
        }

        if (valid_q) {
            float inv_sum = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; ++d) {
                attn_output[(q_idx * (batch * num_heads) + bh) * head_dim + d] = acc_reg[d] * inv_sum;
            }
        }
        __syncthreads();
    }
}

// Average weights kernel: one block per (batch, q_idx), computes softmax per head and averages.
__global__ void average_weights_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ avg_weights,
    int seq_len, int batch, int num_heads, int head_dim,
    float scale) {
    int b = blockIdx.x / seq_len;
    int q_idx = blockIdx.x % seq_len;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Use shared memory for reductions: num_heads * blockDim.x for max and sum
    extern __shared__ float red_smem[];
    float* max_sh = red_smem;
    float* sum_sh = max_sh + num_heads * blockDim.x;

    // Initialize shared to -INFINITY and 0
    for (int h = 0; h < num_heads; ++h) {
        for (int t = threadIdx.x; t < blockDim.x; t += blockDim.x) {
            max_sh[h * blockDim.x + t] = -INFINITY;
            sum_sh[h * blockDim.x + t] = 0.0f;
        }
    }
    __syncthreads();

    // Each thread computes partial max for its subset of keys for each head
    float local_max[32];
    for (int h = 0; h < num_heads; ++h) local_max[h] = -INFINITY;
    for (int k_idx = tid; k_idx < seq_len; k_idx += stride) {
        for (int h = 0; h < num_heads; ++h) {
            float score = 0.0f;
            int bh = b * num_heads + h;
            const float* q_ptr = q + (q_idx * (batch * num_heads) + bh) * head_dim;
            const float* k_ptr = k + (k_idx * (batch * num_heads) + bh) * head_dim;
            for (int d = 0; d < head_dim; ++d) score += q_ptr[d] * k_ptr[d];
            score *= scale;
            if (score > local_max[h]) local_max[h] = score;
        }
    }
    // Write partial max to shared
    for (int h = 0; h < num_heads; ++h) {
        max_sh[h * blockDim.x + tid] = local_max[h];
    }
    __syncthreads();

    // Reduce max across threads for each head
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            for (int h = 0; h < num_heads; ++h) {
                if (max_sh[h * blockDim.x + tid + offset] > max_sh[h * blockDim.x + tid])
                    max_sh[h * blockDim.x + tid] = max_sh[h * blockDim.x + tid + offset];
            }
        }
        __syncthreads();
    }
    float global_max[32];
    for (int h = 0; h < num_heads; ++h) global_max[h] = max_sh[h * blockDim.x];

    // Compute partial sums of exp(score - max)
    float local_sum[32];
    for (int h = 0; h < num_heads; ++h) local_sum[h] = 0.0f;
    for (int k_idx = tid; k_idx < seq_len; k_idx += stride) {
        for (int h = 0; h < num_heads; ++h) {
            float score = 0.0f;
            int bh = b * num_heads + h;
            const float* q_ptr = q + (q_idx * (batch * num_heads) + bh) * head_dim;
            const float* k_ptr = k + (k_idx * (batch * num_heads) + bh) * head_dim;
            for (int d = 0; d < head_dim; ++d) score += q_ptr[d] * k_ptr[d];
            score *= scale;
            local_sum[h] += expf(score - global_max[h]);
        }
    }
    // Write partial sums to shared
    for (int h = 0; h < num_heads; ++h) {
        sum_sh[h * blockDim.x + tid] = local_sum[h];
    }
    __syncthreads();
    // Reduce sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            for (int h = 0; h < num_heads; ++h) {
                sum_sh[h * blockDim.x + tid] += sum_sh[h * blockDim.x + tid + offset];
            }
        }
        __syncthreads();
    }
    float global_sum[32];
    for (int h = 0; h < num_heads; ++h) global_sum[h] = sum_sh[h * blockDim.x];

    // Each thread computes probability for its key indices and averages over heads, writes to avg_weights
    for (int k_idx = tid; k_idx < seq_len; k_idx += stride) {
        float prob_sum = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            float score = 0.0f;
            int bh = b * num_heads + h;
            const float* q_ptr = q + (q_idx * (batch * num_heads) + bh) * head_dim;
            const float* k_ptr = k + (k_idx * (batch * num_heads) + bh) * head_dim;
            for (int d = 0; d < head_dim; ++d) score += q_ptr[d] * k_ptr[d];
            score *= scale;
            float prob = expf(score - global_max[h]) / global_sum[h];
            prob_sum += prob;
        }
        avg_weights[(b * seq_len + q_idx) * seq_len + k_idx] = prob_sum / num_heads;
    }
}

// Concatenate along last dim
__global__ void cat_last_dim_kernel(
    const float* __restrict__ t1,
    const float* __restrict__ t2,
    float* __restrict__ output,
    int outer_size,
    int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = outer_size * 2 * inner_size;
    for (int i = idx; i < total; i += stride) {
        int outer = i / (2 * inner_size);
        int inner_half = i % (2 * inner_size);
        int src_inner = (inner_half < inner_size) ? inner_half : inner_half - inner_size;
        const float* src = (inner_half < inner_size) ? t1 : t2;
        output[i] = src[outer * inner_size + src_inner];
    }
}

// GLU along last dim
__global__ void glu_last_dim_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int outer_size,
    int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = outer_size * inner_size;
    for (int i = idx; i < total; i += stride) {
        int outer = i / inner_size;
        int inner = i % inner_size;
        float a = x[outer * 2 * inner_size + inner];
        float b = x[outer * 2 * inner_size + inner_size + inner];
        output[i] = a * sigmoidf(b);
    }
}

// Linear kernel
__global__ void linear_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int outer_size,
    int in_features,
    int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = outer_size * out_features;
    for (int i = idx; i < total; i += stride) {
        int row = i / out_features;
        int col = i % out_features;
        float sum = 0.0f;
        const float* x_row = x + row * in_features;
        const float* w_row = weight + col * in_features;
        for (int d = 0; d < in_features; ++d) {
            sum += x_row[d] * w_row[d];
        }
        if (bias != nullptr) sum += bias[col];
        output[i] = sum;
    }
}

// Host launchers
extern "C" void custom_multihead_attention_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* avg_weights,
    int seq_len, int batch, int embed_dim, int num_heads,
    bool need_weights, bool average_attn_weights,
    cudaStream_t stream) {
    int head_dim = embed_dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate temporary q,k,v
    float *d_q, *d_k, *d_v;
    size_t qkv_size = (size_t)seq_len * batch * num_heads * head_dim * sizeof(float);
    cudaMalloc(&d_q, qkv_size);
    cudaMalloc(&d_k, qkv_size);
    cudaMalloc(&d_v, qkv_size);

    // Launch in_projection_split_kernel
    int total_proj = seq_len * batch * 3 * embed_dim;
    int proj_threads = 256;
    int proj_blocks = (total_proj + proj_threads - 1) / proj_threads;
    in_projection_split_kernel<<<proj_blocks, proj_threads, 0, stream>>>(
        query, key, value, in_proj_weight, in_proj_bias,
        d_q, d_k, d_v, seq_len, batch, embed_dim, num_heads);

    // Launch attention_output_kernel
    int attn_blocks = batch * num_heads;
    int tile_threads = 32;  // must be <= 1024 and match TILE_Q/TILE_K
    // shared mem: K_tile (TILE_K*head_dim) + V_tile (TILE_K*head_dim) + q_tile (TILE_Q*head_dim) = 3 * 32 * head_dim
    size_t attn_shared = (size_t)3 * tile_threads * head_dim * sizeof(float);
    attention_output_kernel<<<attn_blocks, tile_threads, attn_shared, stream>>>(
        d_q, d_k, d_v, attn_output,
        seq_len, batch, num_heads, head_dim, scale);

    // Launch average_weights_kernel if needed
    if (need_weights && average_attn_weights) {
        int avg_blocks = batch * seq_len;
        int avg_threads = 256;
        size_t avg_shared = (size_t)num_heads * avg_threads * 2 * sizeof(float);
        average_weights_kernel<<<avg_blocks, avg_threads, avg_shared, stream>>>(
            d_q, d_k, avg_weights,
            seq_len, batch, num_heads, head_dim, scale);
    }

    // Out projection: attn_output (seq_len, batch, embed_dim) = attn_output @ out_proj_weight^T + out_proj_bias
    // attn_output is currently (seq_len, batch*num_heads, head_dim) contiguous = (seq_len, batch, embed_dim)
    int outer_size = seq_len * batch;
    int linear_threads = 256;
    int linear_blocks = (outer_size * embed_dim + linear_threads - 1) / linear_threads;
    linear_kernel<<<linear_blocks, linear_threads, 0, stream>>>(
        attn_output, out_proj_weight, out_proj_bias, attn_output,
        outer_size, embed_dim, embed_dim);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
}

extern "C" void custom_cat_launcher(
    const float* t1, const float* t2, float* output,
    int outer_size, int inner_size, cudaStream_t stream) {
    int total = outer_size * 2 * inner_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cat_last_dim_kernel<<<blocks, threads, 0, stream>>>(t1, t2, output, outer_size, inner_size);
}

extern "C" void custom_glu_launcher(
    const float* x, float* output,
    int outer_size, int inner_size, cudaStream_t stream) {
    int total = outer_size * inner_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    glu_last_dim_kernel<<<blocks, threads, 0, stream>>>(x, output, outer_size, inner_size);
}

extern "C" void custom_linear_launcher(
    const float* x, const float* weight, const float* bias, float* output,
    int outer_size, int in_features, int out_features, cudaStream_t stream) {
    int total = outer_size * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    linear_kernel<<<blocks, threads, 0, stream>>>(x, weight, bias, output, outer_size, in_features, out_features);
}
