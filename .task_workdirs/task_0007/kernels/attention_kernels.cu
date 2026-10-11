#include <cuda_runtime.h>
#include <cfloat>

// ============================================================================
// QKV Projection + Positional Encoding Kernel
// Computes: q = query @ W_q^T + pos_enc + bq, similar for k, v
// Inputs: query, key, value: [B, T, E]
//         pos_enc: [1, P, E]
//         wq, wk, wv: [E, E] row-major (PyTorch Linear weight)
//         bq, bk, bv: optional [E]
// Outputs: q_out, k_out, v_out: [B, T, E]
// ============================================================================
template<int BLOCK_SIZE>
__global__ void qkv_proj_pos_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ pos_enc,
    const float* __restrict__ wq,
    const float* __restrict__ wk,
    const float* __restrict__ wv,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv,
    int B, int T, int E, int P,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
    // One block per batch and time step, threads across E
    int batch = blockIdx.x;
    int t = blockIdx.y;
    if (batch >= B || t >= T) return;

    int tid = threadIdx.x;
    int idx = (batch * T + t) * E + tid;

    if (tid < E) {
        // Load pos_enc once per position (assumes P >= T)
        float pe = pos_enc[t * E + tid];

        // Compute q, k, v linear projections + bias + pos_enc
        float sum_q = (bq != nullptr) ? bq[tid] : 0.0f;
        float sum_k = (bk != nullptr) ? bk[tid] : 0.0f;
        float sum_v = (bv != nullptr) ? bv[tid] : 0.0f;

        for (int e = 0; e < E; ++e) {
            float q_val = query[batch * T * E + t * E + e];
            float k_val = key[batch * T * E + t * E + e];
            float v_val = value[batch * T * E + t * E + e];
            sum_q += q_val * wq[e * E + tid];
            sum_k += k_val * wk[e * E + tid];
            sum_v += v_val * wv[e * E + tid];
        }

        q_out[idx] = sum_q + pe;
        k_out[idx] = sum_k + pe;
        v_out[idx] = sum_v + pe;
    }
}

// ============================================================================
// Scaled Dot-Product Attention (SDPA) Kernel
// Computes: out = softmax(QK^T / sqrt(E) + mask) V
// Q, K, V: [B, T, E], mask: [T, T] (broadcast over batch)
// Output: [B, T, E]
// ============================================================================
template<int BLOCK_SIZE>
__global__ void sdpa_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ mask,
    int B, int T, int E,
    float* __restrict__ out
) {
    // Each block computes one attention row (batch, row)
    int batch = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;

    if (batch >= B || row >= T) return;

    // Load q row into shared memory
    extern __shared__ float smem[];
    float* s_q = smem;              // [E]
    float* s_k = &s_q[E];           // [E]
    float* s_v = &s_k[E];           // [E] (will be placed after s_k)

    for (int e = tid; e < E; e += BLOCK_SIZE) {
        s_q[e] = q[(batch * T + row) * E + e];
    }
    __syncthreads();

    // Initialize accumulator for output elements (each thread handles one element if tid < E)
    float acc = 0.0f;
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;

    for (int j = 0; j < T; ++j) {
        // Load K_j and V_j into shared memory
        for (int e = tid; e < E; e += BLOCK_SIZE) {
            s_k[e] = k[(batch * T + j) * E + e];
            s_v[e] = v[(batch * T + j) * E + e];
        }
        __syncthreads();

        // Compute dot product Q·K_j (all threads participate)
        float dot = 0.0f;
        for (int e = 0; e < E; ++e) {
            dot += s_q[e] * s_k[e];
        }
        dot /= sqrtf((float)E);

        // Add mask if provided (broadcast over batch)
        if (mask != nullptr) {
            dot += mask[row * T + j];
        }

        // Online softmax update
        float new_max = fmaxf(max_score, dot);
        float correction = __expf(max_score - new_max);
        sum_exp = sum_exp * correction + __expf(dot - new_max);
        max_score = new_max;

        // Accumulate V_j * exp(dot - max_score) for this thread's output element
        float weight = __expf(dot - max_score);
        if (tid < E) {
            acc = acc * correction + weight * s_v[tid];
        }
        __syncthreads();
    }

    // Normalize and write out
    if (tid < E) {
        out[(batch * T + row) * E + tid] = acc / sum_exp;
    }
}

// ============================================================================
// Attention Output Times Its Transpose (baddbmm with beta=0)
// Computes: logits = attn_output @ attn_output^T, [B, T, E] -> [B, T, T]
// ============================================================================
template<int BLOCK_SIZE>
__global__ void attn_bias_baddbmm_kernel(
    const float* __restrict__ attn_output,
    int B, int T, int E,
    float* __restrict__ logits
) {
    int batch = blockIdx.x;
    int row = blockIdx.y;
    int col = blockIdx.z;

    if (batch >= B || row >= T || col >= T) return;

    int tid = threadIdx.x;

    // One block computes one output element (row, col) by summing over E
    float sum = 0.0f;
    for (int e = tid; e < E; e += BLOCK_SIZE) {
        sum += attn_output[(batch * T + row) * E + e] *
               attn_output[(batch * T + col) * E + e];
    }

    // Reduce within block
    __shared__ float s_partials[BLOCK_SIZE];
    s_partials[tid] = sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_partials[tid] += s_partials[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        logits[(batch * T + row) * T + col] = s_partials[0];
    }
}

// ============================================================================
// BCE With Logits Loss Kernel
// Computes: loss = mean over all elements of max(x,0) - x*y + log(1+exp(-|x|))
// ============================================================================
template<int BLOCK_SIZE>
__global__ void bce_with_logits_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ target,
    int B, int T,
    float* __restrict__ loss
) {
    int N = B * T * T;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int idx = i; idx < N; idx += gridDim.x * blockDim.x) {
        float x = logits[idx];
        float y = target[idx];
        // Stable BCE with logits: max(x,0) - x*y + log(1+exp(-abs(x)))
        float loss_val = fmaxf(x, 0.0f) - x * y + log1pf(expf(-fabsf(x)));
        sum += loss_val;
    }

    // Block reduction
    __shared__ float s_partials[BLOCK_SIZE];
    int tid = threadIdx.x;
    s_partials[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_partials[tid] += s_partials[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, s_partials[0] / N);
    }
}

// ============================================================================
// Launcher functions (extern "C")
// ============================================================================
extern "C" void qkv_proj_pos_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* pos_enc,
    const float* wq,
    const float* wk,
    const float* wv,
    const float* bq,
    const float* bk,
    const float* bv,
    int B,
    int T,
    int E,
    int P,
    float* q_out,
    float* k_out,
    float* v_out,
    cudaStream_t stream
) {
    dim3 blocks(B, T);
    int threads = 64; // E=64, use 64 threads
    qkv_proj_pos_kernel<64><<<blocks, threads, 0, stream>>>(
        query, key, value, pos_enc, wq, wk, wv, bq, bk, bv,
        B, T, E, P, q_out, k_out, v_out
    );
}

extern "C" void sdpa_forward_launcher(
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    int B,
    int T,
    int E,
    float* out,
    cudaStream_t stream
) {
    // Use one block per (batch, row), 64 threads
    dim3 blocks(B, T);
    int threads = 64;
    int shared_mem = 3 * E * sizeof(float); // q, k, v in shared
    sdpa_forward_kernel<64><<<blocks, threads, shared_mem, stream>>>(
        q, k, v, mask, B, T, E, out
    );
}

extern "C" void attn_bias_baddbmm_launcher(
    const float* attn_output,
    int B,
    int T,
    int E,
    float* logits,
    cudaStream_t stream
) {
    // One block per output element, 64 threads
    dim3 blocks(B, T, T);
    int threads = 64;
    attn_bias_baddbmm_kernel<64><<<blocks, threads, 0, stream>>>(
        attn_output, B, T, E, logits
    );
}

extern "C" void bce_with_logits_launcher(
    const float* logits,
    const float* target,
    int B,
    int T,
    float* loss,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (B * T * T + threads - 1) / threads;
    // Initialize loss to 0 before reduction
    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    bce_with_logits_kernel<256><<<blocks, threads, 0, stream>>>(
        logits, target, B, T, loss
    );
}
