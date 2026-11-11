// Custom CUDA kernels for multihead attention and matmul
// Only CUDA kernels and launchers, no torch/ATen/c10 dependencies.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Helper: linear projection (matrix multiply) for query/key/value
// Computes out[b,t,d] = sum_k in[b,t,k] * weight[d,k] + bias[d]
// in shape: [B, T, C] (contiguous or not, handled via strides)
// weight shape: [D, C]
// bias shape: [D]
// out shape: [B, T, D]
// This kernel assumes out is contiguous, but input may have arbitrary strides.
// We use standard row-major indexing but respect strides for input.
// For simplicity and performance, we require input to be contiguous (binding will ensure).
// However, to be safe, we handle strides via parameters but use contiguous memory for speed.
// Since contract says supports strides, binding will call .contiguous() if needed.
// The launcher will receive raw pointers and strides.
// We implement a simple grid-stride loop over all output elements.
// Each thread computes one output element by doing a dot product.

// Macro for checking CUDA errors (only used in launchers, not inside kernels)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// -----------------------------------------------------------------------------
// Linear projection kernel: out = in * weight^T + bias
// in: [B, T, C] with strides: in_b_stride, in_t_stride, in_c_stride (in elements)
// weight: [D, C] row-major
// bias: [D]
// out: [B, T, D] assumed contiguous (out_b_stride = T*D, out_t_stride = D, out_d_stride = 1)
// Each thread handles one output element (b,t,d)
__global__ void linear_projection_kernel(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int T, int C, int D,
    int in_b_stride, int in_t_stride, int in_c_stride,
    int out_b_stride, int out_t_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * D;
    if (idx >= total) return;
    int d = idx % D;
    int tmp = idx / D;
    int t = tmp % T;
    int b = tmp / T;

    const float* in_row = in + b * in_b_stride + t * in_t_stride;
    const float* w_row = weight + d * C;
    float sum = bias[d];
    for (int c = 0; c < C; ++c) {
        sum += in_row[c * in_c_stride] * w_row[c];
    }
    out[b * out_b_stride + t * out_t_stride + d] = sum;
}

// -----------------------------------------------------------------------------
// Scaled dot-product attention with multi-head splitting
// Inputs: q, k, v after projection, each shape [B, T, D] where D = embed_dim
// num_heads, head_dim = D / num_heads
// q, k, v are used to compute attention scores and output.
// We will compute attention for all heads in parallel.
// attn_weights is output of shape [B, T, T] only if need_weights.
// But contract only has need_weights=True, so we always compute and store attn_weights.
// The actual attention output is stored in out: [B, T, D]
// We use a tile-based approach for efficiency, but for simplicity we can have each thread
// compute one output element of the attention output (b, t, d) and also compute
// one row of attention weights? That would be slow but correct.
// Better: use a kernel that computes attention weights for all heads for a given query position.
// Since T=1024, B=32, D=128, num_heads=4, head_dim=32.
// We can launch a kernel with grid = (B * num_heads * T) / block.x? Each block computes
// one query position and head, over all key positions.
// Shared memory can be used to store q and k for the head.
// But for simplicity first, implement a baseline correct version.

// This kernel computes the attention output for one query position, across all key positions,
// for a specific batch and head, and writes the corresponding output values for that head.
// It also optionally writes the attention weights for that head (but we need average heads later).
// Since average_attn_weights=True, we need to average over heads. 
// But contract says always average_attn_weights=True. So we can compute per-head weights and average,
// or compute directly by summing over heads. The output attn_weights is averaged over heads.
// To produce attn_weights of shape [B, T, T], we need to average.
// Approach: First compute attention output for all heads, and also accumulate attention weights
// into a temporary buffer of shape [B, T, T] initialized to zero. Then after all heads done,
// divide by num_heads to get averaged attention weights. But we must also produce output.
// Alternatively, compute averaged attention weights first, then use those for output (since output uses averaged weights? No, output uses per-head softmax and then concat).
// Actually, MultiheadAttention computes per-head attention and then concatenates heads, not average weights.
// The output is computed using attention weights per head, then out_proj mixes all heads.
// So we cannot simply average weights before output. We need per-head softmax.
// Therefore, the attention weights returned are averaged over heads for each (b,t,s) position (if average_attn_weights=True).
// So we must compute all per-head attentions, store them maybe, then average for output weights tensor.
// To avoid storing all per-head weights (B*num_heads*T*T could be large, but manageable: 32*4*1024*1024=134M floats, ~536MB, maybe okay but not desirable).
// We can compute output and average weights in one kernel: for each (b,t), compute for each head the softmax over keys, and for each head write output contribution, and also accumulate average weight contribution.
// However, each head output for each position t contributes to output at (b,t, specific head dims). So we can have a kernel that for a given (b,t) and a range of heads computes the output and also accumulates
// per-head weights into a per-head weight buffer, then later average? That would require synchronization across heads.
// Better: Use a separate kernel to compute attn_weights (averaged) after computing per-head weights, but we don't need to store per-head weights if we use atomic adds into output attn_weights? Atomic adds are not stable order but addition is commutative, so okay if we only average, no ordering needed.
// But atomicAdd on float is fine, and we don't need stable order for sum. So we can initialize attn_weights to zero, then for each (b,t,head,s) compute softmax value, atomicAdd to attn_weights[b,t,s] / num_heads? But atomicAdd with division later? We can accumulate and then divide by num_heads in a separate small kernel.
// That is feasible: for each head, compute attention scores, softmax, then add attention weight to attn_weights (divided by num_heads? we can add raw weight and later divide). Also compute output contribution for that head: out[b,t, head*head_dim + d] = sum_s attn_{head}(s) * v[b,s, head*head_dim + d].
// So we need a kernel that for each (b,t,head) computes the entire softmax over S=seq_len keys, then writes output for that head, and atomicAdd to attn_weights.
// But atomicAdd on attn_weights could be slow due to contention, but S=1024, B=32, heads=4, total atomic adds = 32*1024*4*1024 = 134M, might be okay but not ideal.
// Alternative: since we have a separate buffer for per-head weights (shape [B, num_heads, T, T]), we can compute per-head weights, then average later. But per-head weights buffer size is same as average, just extra num_heads dimension. That's 32*4*1024*1024*4 bytes = 536MB. Might be too much but maybe okay? Better to avoid.
// Another approach: compute output without storing per-head weights by fusing softmax and output in a single kernel that processes heads one by one for each (b,t). That kernel would be complex.
// For simplicity and correctness, we can implement a kernel that computes attention for each head and writes both output and per-head weights to separate buffers, then have a second kernel to average per-head weights into the final attn_weights. Memory overhead of per-head weights: B*H*T*T floats = 32*4*1M*4 = 536MB. That might be too much if GPU memory limited, but for current problem maybe okay (modern GPUs have >8GB). However, to be safe, we can use a different scheme: compute per-head attention weights on the fly for output and simultaneously accumulate averaged weights using atomicAdd into a zero-initialized attn_weights. Since atomicAdd is deterministic in sum (floating-point addition order not fixed but acceptable for tolerance), we can do that. We'll use atomicAdd for attn_weights accumulation, and after all heads, divide by num_heads.
// So kernel 1: compute attention scores for each head, softmax, accumulate into attn_weights (raw sum over heads), and also compute output for that head (write directly to out, as different heads write to different output dimensions, no conflict).
// Then kernel 2: divide attn_weights by num_heads.
// But we must be careful: out has shape [B, T, D], and each head h writes to dimensions [h*head_dim, (h+1)*head_dim). No conflict between heads. Within a head, for a given b,t, all threads writing output for that head are within a block? We'll have one block per (b,t,head), computing the entire attention row for that head. We'll use shared memory to store q,k,v for that head to avoid repeated loads.
// Since head_dim=32, T=1024, we can have blockDim.x = 256 (or 128). Each block computes attention for one (b,t,head).
// GridDim.x = B * T * num_heads. That's 32*1024*4 = 131072 blocks, each with e.g. 256 threads. That's 33M threads, okay.
// Within a block, we compute attention scores for all keys: each thread handles multiple keys, then softmax using online softmax or reduction.
// Simpler: each block uses shared memory array of size T to store scores, but T=1024, that's 4KB per block, okay. But we have 131072 blocks; each block uses 4KB shared, that's 512MB shared memory total across all SMs? That's too much! Shared memory is per SM; each SM can run multiple blocks concurrently, so occupancy might be hurt. Better use a smaller tile size and compute softmax in chunks.
// But for correctness first, we might use a naive but correct kernel: each thread in block computes a subset of scores, then use block reduction to compute max and sum, then compute softmax and output.
// We can use shared memory of size blockDim.x for reduction, not T. We'll have each thread compute partial scores for its assigned keys, then find max via block reduction, then compute sum of exp via block reduction, then each thread calculates its portion of output and writes.
// This requires multiple passes over keys. Since T=1024 and blockDim=256, each thread handles 4 keys.
// We'll implement a kernel where each block handles one (b,t,head), and threads cooperate via shared memory reduction.
// This will be correct and reasonably efficient.

// Definitions for attention kernel:
// q_proj: [B, T, D], k_proj: [B, T, D], v_proj: [B, T, D]
// out: [B, T, D], attn_weights: [B, T, T] (accumulator)
// head_dim = D / num_heads
// d_k = head_dim
// scaling = 1/sqrt(head_dim)

__global__ void attention_kernel(
    const float* __restrict__ q_proj,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    float* __restrict__ out,
    float* __restrict__ attn_weights, // accumulated over heads, not yet averaged
    int B, int T, int D, int num_heads, int head_dim,
    float scaling
) {
    int block_idx = blockIdx.x; // 0..B*T*num_heads-1
    int h = block_idx % num_heads;
    int tmp = block_idx / num_heads;
    int t = tmp % T;
    int b = tmp / T;

    // Pointers to the query vector for this (b,t) and head
    const float* q = q_proj + ((long long)b * T + t) * D + h * head_dim;
    const float* k_base = k_proj + (long long)b * T * D + h * head_dim; // index of (b,0,h,0)
    const float* v_base = v_proj + (long long)b * T * D + h * head_dim;
    float* out_base = out + ((long long)b * T + t) * D + h * head_dim;

    // Each block handles one query position and head.
    // We need to compute softmax over all T keys.
    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Step 1: find max score
    float max_score = -1e30f;
    for (int s = tid; s < T; s += threads) {
        const float* k = k_base + s * D;
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += q[d] * k[d];
        }
        score *= scaling;
        if (score > max_score) max_score = score;
    }

    // Block reduce max
    __shared__ float s_max[256]; // assume threads <= 256, later check
    s_max[tid] = max_score;
    __syncthreads();
    for (int stride = threads/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = max(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = s_max[0];

    // Step 2: compute sum of exp
    float sum_exp = 0.0f;
    for (int s = tid; s < T; s += threads) {
        const float* k = k_base + s * D;
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += q[d] * k[d];
        }
        score = (score * scaling) - global_max;
        sum_exp += __expf(score);
    }

    // Block reduce sum
    __shared__ float s_sum[256];
    s_sum[tid] = sum_exp;
    __syncthreads();
    for (int stride = threads/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = s_sum[0];

    // Step 3: compute output and accumulate attention weights
    // For each key s, compute attention weight = exp(score - max)/sum
    // For output, for each d in head_dim, out[d] = sum_s weight_s * v_s[d]
    // Also accumulate weight_s into attn_weights[b,t,s] using atomicAdd
    for (int s = tid; s < T; s += threads) {
        const float* k = k_base + s * D;
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += q[d] * k[d];
        }
        score = __expf((score * scaling) - global_max) / global_sum;
        float weight = score; // attention probability for this (b,t,head,s)
        // Accumulate into attn_weights (we'll divide by num_heads later)
        atomicAdd(&attn_weights[((long long)b * T + t) * T + s], weight);

        // Compute output contribution: out[d] += weight * v_s[d]
        const float* v = v_base + s * D;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            atomicAdd(&out_base[d], weight * v[d]);
        }
    }
}

// Kernel to divide attn_weights by num_heads after accumulation
__global__ void average_attn_weights_kernel(float* attn_weights, float inv_num_heads, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        attn_weights[idx] *= inv_num_heads;
    }
}

// -----------------------------------------------------------------------------
// Matmul kernel for 3D batch multiplication: out[b,i,j] = sum_k a[b,i,k] * b[b,k,j]
// a: [B, M, K], b: [B, K, N], out: [B, M, N]
// Strides may vary; we assume contiguous for a and out, but b could be non-contiguous (transpose)
// However contract says b is value.transpose(1,2) which is contiguous in memory but shape [B, embed_dim, seq_len].
// So we can treat b as contiguous with shape [B, K, N] and stride K*N, N, 1.
// We'll implement a simple kernel: each thread computes one output element.
__global__ void batched_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int B, int M, int K, int N,
    int a_b_stride, int a_m_stride, int a_k_stride,
    int b_b_stride, int b_k_stride, int b_n_stride,
    int out_b_stride, int out_m_stride, int out_n_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * M * N;
    if (idx >= total) return;
    int n = idx % N;
    int tmp = idx / N;
    int m = tmp % M;
    int b = tmp / M;

    float sum = 0.0f;
    const float* a_row = a + b * a_b_stride + m * a_m_stride;
    const float* b_col = b + b * b_b_stride + n * b_n_stride;
    for (int k = 0; k < K; ++k) {
        sum += a_row[k * a_k_stride] * b_col[k * b_k_stride];
    }
    out[b * out_b_stride + m * out_m_stride + n * out_n_stride] = sum;
}

// -----------------------------------------------------------------------------
// Launchers (extern "C")

// Launcher for multihead attention forward
// Parameters are raw pointers, dimensions, strides, and bool flags.
// The binding code will ensure all tensors are contiguous and on CUDA, and convert optional masks appropriately.
// We require that key_padding_mask and attn_mask are either null or preprocessed by binding to be simple tensors; 
// but for this task, they are always null, so we ignore them, but include parameters for interface compatibility.
// For simplicity, we assume null and non-null not passed.
// The launcher performs:
// 1. Allocate temporary buffers for projected q,k,v (size B*T*D each) using cudaMalloc.
// 2. Launch linear_projection_kernel for each projection.
// 3. Allocate attn_weights_temp (size B*T*T) initialized to zero for accumulation.
// 4. Launch attention_kernel for each block (one per b,t,h).
// 5. Launch average_attn_weights_kernel to divide by num_heads.
// 6. Launch linear_projection_kernel for out_proj (input is attention output, output is final attn_output).
// 7. Free temporaries.
// Note: out_proj weight shape [embed_dim, embed_dim] and bias [embed_dim]. 
// Attention output before out_proj has shape [B, T, D], same as projected query.
// We need to apply out_proj linearly.
// We also need to produce attn_weights output with shape [B, T, T] after averaging.
// All allocations use cudaMalloc; binding will allocate output tensors and pass pointers.
// The launcher returns void; errors should be checked by binding after launch via cudaGetLastError, but we also check inside.

extern "C" void multihead_attention_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* attn_weights,
    int B, int T, int E, int num_heads,
    bool batch_first, bool add_zero_attn, bool is_causal,
    const float* key_padding_mask, const float* attn_mask,
    bool need_weights, bool average_attn_weights,
    cudaStream_t stream
) {
    // In this task, batch_first is always true, add_zero_attn=false, is_causal=false.
    // We assume batch_first true. If not, we would need to transpose, but binding ensures true.
    // We ignore key_padding_mask and attn_mask as they are null in this task.
    // We also ignore add_zero_attn, is_causal.

    int D = E; // embed_dim
    int head_dim = D / num_heads;
    float scaling = 1.0f / sqrtf((float)head_dim);

    // Temporary buffers for projected q,k,v
    float *q_proj, *k_proj, *v_proj;
    size_t proj_size = (size_t)B * T * D * sizeof(float);
    cudaMalloc(&q_proj, proj_size);
    cudaMalloc(&k_proj, proj_size);
    cudaMalloc(&v_proj, proj_size);

    // Temporary buffer for accumulated attention weights before averaging
    float* attn_weights_acc;
    size_t attn_size = (size_t)B * T * T * sizeof(float);
    cudaMalloc(&attn_weights_acc, attn_size);
    cudaMemsetAsync(attn_weights_acc, 0, attn_size, stream);

    // Output for attention before out_proj (same as q_proj size)
    float* attn_out_tmp;
    cudaMalloc(&attn_out_tmp, proj_size);

    // Launch linear projections for q,k,v
    // Input query, key, value are assumed contiguous with shape [B,T,E]
    // We'll set strides accordingly.
    int in_b_stride = T * E;
    int in_t_stride = E;
    int in_c_stride = 1;
    int out_b_stride = T * D;
    int out_t_stride = D;

    // Query projection -> q_proj [B,T,D]
    {
        int total_threads = B * T * D;
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        linear_projection_kernel<<<blocks, threads, 0, stream>>>(
            query, in_proj_weight, in_proj_bias, q_proj,
            B, T, E, D,
            in_b_stride, in_t_stride, in_c_stride,
            out_b_stride, out_t_stride
        );
    }
    // Key projection -> k_proj
    {
        int total_threads = B * T * D;
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        linear_projection_kernel<<<blocks, threads, 0, stream>>>(
            key, in_proj_weight, in_proj_bias, k_proj,
            B, T, E, D,
            in_b_stride, in_t_stride, in_c_stride,
            out_b_stride, out_t_stride
        );
    }
    // Value projection -> v_proj
    {
        int total_threads = B * T * D;
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        linear_projection_kernel<<<blocks, threads, 0, stream>>>(
            value, in_proj_weight, in_proj_bias, v_proj,
            B, T, E, D,
            in_b_stride, in_t_stride, in_c_stride,
            out_b_stride, out_t_stride
        );
    }

    // Launch attention kernel
    int total_blocks = B * T * num_heads;
    int attn_threads = 256; // block size
    attention_kernel<<<total_blocks, attn_threads, 0, stream>>>(
        q_proj, k_proj, v_proj,
        attn_out_tmp, attn_weights_acc,
        B, T, D, num_heads, head_dim, scaling
    );

    // Average attention weights if average_attn_weights and need_weights
    if (need_weights && average_attn_weights) {
        int total_attn = B * T * T;
        int threads = 256;
        int blocks = (total_attn + threads - 1) / threads;
        average_attn_weights_kernel<<<blocks, threads, 0, stream>>>(
            attn_weights_acc, 1.0f / num_heads, total_attn
        );
    }

    // Apply out_proj: attn_output = attn_out_tmp * out_proj_weight^T + out_proj_bias
    // out_proj_weight shape [D, D], we can reuse linear_projection_kernel with D==out_dim, C==D.
    {
        int total_threads = B * T * D;
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        linear_projection_kernel<<<blocks, threads, 0, stream>>>(
            attn_out_tmp, out_proj_weight, out_proj_bias, attn_output,
            B, T, D, D,
            out_b_stride, out_t_stride, 1, // input strides same as output of attention
            out_b_stride, out_t_stride // output strides for attn_output (assumed contiguous)
        );
    }

    // Copy attn_weights_acc to attn_weights output (if need_weights)
    if (need_weights) {
        cudaMemcpyAsync(attn_weights, attn_weights_acc, attn_size, cudaMemcpyDeviceToDevice, stream);
    }

    // Free temporaries (asynchronously? cudaFree is fine, but must ensure kernels complete? We can rely on stream execution)
    cudaFree(q_proj);
    cudaFree(k_proj);
    cudaFree(v_proj);
    cudaFree(attn_weights_acc);
    cudaFree(attn_out_tmp);

    // Note: we may want to check errors after kernel launches, but binding will check cudaGetLastError.
    // We can add a check here as well.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error? We cannot throw, but we can maybe print.
        // Since this is a C interface, we can't easily report, but binding will check after call.
        // We'll do nothing.
    }
}

// Launcher for batched matmul
// a: [B, M, K], b: [B, K, N], out: [B, M, N]
// All are assumed contiguous, but we'll pass strides to be safe.
// In this task, a is from attention output (B,T,E) contiguous, b is value.transpose(1,2) which is contiguous with shape [B,E,T], strides [E*T, T, 1].
// So we can use direct strides.
extern "C" void matmul_forward_launcher(
    const float* a, const float* b, float* out,
    int B, int M, int K, int N,
    int a_b_stride, int a_m_stride, int a_k_stride,
    int b_b_stride, int b_k_stride, int b_n_stride,
    int out_b_stride, int out_m_stride, int out_n_stride,
    cudaStream_t stream
) {
    int total = B * M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    batched_matmul_kernel<<<blocks, threads, 0, stream>>>(
        a, b, out,
        B, M, K, N,
        a_b_stride, a_m_stride, a_k_stride,
        b_b_stride, b_k_stride, b_n_stride,
        out_b_stride, out_m_stride, out_n_stride
    );
}
