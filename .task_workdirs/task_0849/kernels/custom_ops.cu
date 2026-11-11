#include <cuda_runtime.h>
#include <cstdint>

// Helper: 1D grid-stride elementwise kernel
template<typename scalar_t>
__global__ void copy_kernel(const scalar_t* __restrict__ src, scalar_t* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// Compute q, k, v projections: q = query @ Wq^T + bq, etc.
// query, key, value: (L, N, E) non-contiguous possibly, Wq, Wk, Wv: (E, D) contiguous, bias: (D) contiguous
// output: (L, N, D) contiguous
template<typename scalar_t>
__global__ void linear_projection_kernel(
    const scalar_t* __restrict__ input,    // (L*N, E) if viewed as 2D, but we use effective strides
    const scalar_t* __restrict__ weight,   // (E, D)
    const scalar_t* __restrict__ bias,     // (D)
    scalar_t* __restrict__ output,         // (L, N, D) contiguous
    int64_t L, int64_t N, int64_t E, int64_t D,
    int64_t input_row_stride,   // stride between consecutive L rows (N*E for contiguous)
    int64_t input_col_stride    // stride between consecutive N indices within a row (E for contiguous)
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total = L * N * D;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t d = i % D;
        int64_t tmp = i / D;
        int64_t n = tmp % N;
        int64_t l = tmp / N;
        const scalar_t* input_row = input + l * input_row_stride + n * input_col_stride;
        scalar_t sum = bias ? bias[d] : scalar_t(0);
        // dot product over E
        for (int64_t e = 0; e < E; ++e) {
            sum += input_row[e] * weight[e * D + d];
        }
        output[i] = sum;
    }
}

// Softmax along last dimension (D) for each (L,N) row
// input, output: contiguous (L*N, D)
template<typename scalar_t>
__global__ void softmax_last_dim_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t rows, int64_t cols
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const scalar_t* in_row = input + row * cols;
    scalar_t* out_row = output + row * cols;
    // find max
    scalar_t max_val = in_row[0];
    for (int64_t j = 1; j < cols; ++j) {
        if (in_row[j] > max_val) max_val = in_row[j];
    }
    // exp and sum
    scalar_t sum = 0;
    for (int64_t j = 0; j < cols; ++j) {
        scalar_t v = expf(in_row[j] - max_val);
        out_row[j] = v;
        sum += v;
    }
    // normalize
    scalar_t inv_sum = scalar_t(1) / sum;
    for (int64_t j = 0; j < cols; ++j) {
        out_row[j] *= inv_sum;
    }
}

// Compute attention output: out = attn_weights @ V for each head, batch
// attn_weights: (L, S) contiguous, V: (S, D) possibly strided, out: (L, D) contiguous
// Here S = L for self-attention, but we keep general S
template<typename scalar_t>
__global__ void matmul_attn_v_kernel(
    const scalar_t* __restrict__ attn_weights,  // (L, S)
    const scalar_t* __restrict__ V,             // (S, D) with row_stride, col_stride
    scalar_t* __restrict__ out,                 // (L, D) contiguous
    int64_t L, int64_t S, int64_t D,
    int64_t V_row_stride, int64_t V_col_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total = L * D;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t d = i % D;
        int64_t l = i / D;
        scalar_t sum = 0;
        for (int64_t s = 0; s < S; ++s) {
            sum += attn_weights[l * S + s] * V[s * V_row_stride + d * V_col_stride];
        }
        out[i] = sum;
    }
}

// Allocate device memory for temporary tensors (helper in launcher)
void launch_copy_1d(const float* src, float* dst, int64_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (n > 0) copy_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void launch_linear_projection(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t L, int64_t N, int64_t E, int64_t D,
    int64_t input_row_stride, int64_t input_col_stride,
    cudaStream_t stream
) {
    int64_t total = L * N * D;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (total > 0) {
        linear_projection_kernel<<<blocks, threads, 0, stream>>>(
            input, weight, bias, output, L, N, E, D, input_row_stride, input_col_stride);
    }
}

void launch_softmax_last_dim(float* data, int64_t rows, int64_t cols, cudaStream_t stream) {
    if (rows > 0) {
        softmax_last_dim_kernel<<<(unsigned int)rows, 1, 0, stream>>>(data, data, rows, cols);
    }
}

// Fully fused attention for a single (head, batch) pair: compute qk^T, scale, softmax, then attn @ V
// q, k, v are contiguous D-dimensional vectors per position; out is D-dimensional
// L is sequence length, D is head_dim
// q_stride, k_stride, v_stride: stride between positions (L dimension)
template<typename scalar_t>
__global__ void attention_head_kernel(
    const scalar_t* __restrict__ q,   // (L, D)
    const scalar_t* __restrict__ k,   // (L, D)
    const scalar_t* __restrict__ v,   // (L, D)
    scalar_t* __restrict__ out,       // (L, D)
    int64_t L, int64_t D,
    int64_t q_stride, int64_t k_stride, int64_t v_stride,
    scalar_t scale
) {
    int64_t row = blockIdx.x; // output position l
    if (row >= L) return;
    const scalar_t* q_row = q + row * q_stride;
    scalar_t* out_row = out + row * D;

    // compute attention scores for this query against all keys
    // We'll use shared memory to cache scores? For simplicity, compute exp directly storing in registers if D small.
    // Use a local array on stack sized by L? L can be large, so we can't. Need to compute exp twice or use shared memory with block covering L positions.
    // Alternative: use a persistent thread per row and loop twice: first find max, then sum, then output.
    // Here we use a single thread per row; this is memory bound but acceptable for small L?
    // L is sequence length (16 in reference but maybe larger); we keep generic.
    // We'll implement two-pass over keys.
    scalar_t max_score = -1e30f;
    for (int64_t j = 0; j < L; ++j) {
        const scalar_t* k_row = k + j * k_stride;
        scalar_t dot = 0;
        for (int64_t d = 0; d < D; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }
    scalar_t sum_exp = 0;
    for (int64_t j = 0; j < L; ++j) {
        const scalar_t* k_row = k + j * k_stride;
        scalar_t dot = 0;
        for (int64_t d = 0; d < D; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot = (dot - max_score) * scale;
        dot = expf(dot);
        sum_exp += dot;
    }
    scalar_t inv_sum = scalar_t(1) / sum_exp;
    // accumulate weighted V: out_row[d] = sum_j softmax_j * V[j,d]
    for (int64_t d = 0; d < D; ++d) {
        scalar_t sum = 0;
        for (int64_t j = 0; j < L; ++j) {
            const scalar_t* k_row = k + j * k_stride;
            scalar_t dot = 0;
            for (int64_t e = 0; e < D; ++e) {
                dot += q_row[e] * k_row[e];
            }
            dot = (dot - max_score) * scale;
            scalar_t weight = expf(dot) * inv_sum;
            sum += weight * v[j * v_stride + d];
        }
        out_row[d] = sum;
    }
}

// launcher for fused attention over all heads and batch
void launch_attention_heads(
    const float* q, const float* k, const float* v,
    float* attn_output,   // (L, N*D) contiguous or strided? We'll output to a contiguous buffer
    int64_t L, int64_t N, int64_t H, int64_t D,
    int64_t q_stride, int64_t k_stride, int64_t v_stride,
    float scale,
    cudaStream_t stream
) {
    // q, k, v are (L, N, H*D) contiguous, but we want per-head views with stride H*D along position and 1 along head_dim.
    // So q_stride should be N*H*D (or actual stride between L positions).
    // We'll launch one block per (batch, head, position), but that may be many blocks (L*N*H).
    // Use a 2D grid: x = position L, y = batch*head.
    dim3 threads(1);
    dim3 blocks(L, N*H);
    if (L <= 0 || N*H <= 0) return;
    // We need to pass per-head pointers; but q is contiguous with layout (L, N, H, D) if we treat correctly.
    // Actually after projection, q is (L, N, H*D). We can compute base offset for each (n,h) and pass to kernel.
    // Simpler: launch kernel that takes full q, k, v and computes per (n,h) subset, but we need stride along head_dim.
    // We'll create a specialized kernel with grid (L, N*H) and compute n,h from blockIdx.y.
    // Since we already have attention_head_kernel expecting pointers to a contiguous (L,D) matrix, we can make a wrapper kernel that handles strides.
    // For simplicity, we'll assume q, k, v are contiguous and layout (L, N, H*D). We'll define new kernel with blockIdx.y for n,h.
    // The attention_head_kernel above takes stride in elements along L dimension; for a given (n,h), q_base = q + n*H*D + h*D, with stride = N*H*D.
    dim3 grid(L, N*H);
    attention_head_strided_kernel<<<grid, threads, 0, stream>>>(
        q, k, v, attn_output, L, N, H, D, scale);
}

// Strided version: each block computes one output position for a specific (n,h)
template<typename scalar_t>
__global__ void attention_head_strided_kernel(
    const scalar_t* __restrict__ q,  // (L, N, H*D)
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,      // (L, N, H*D) contiguous
    int64_t L, int64_t N, int64_t H, int64_t D,
    scalar_t scale
) {
    int64_t l = blockIdx.x;
    int64_t nh = blockIdx.y;
    int64_t n = nh / H;
    int64_t h = nh % H;
    if (l >= L) return;
    // base offset for this batch, head
    int64_t head_offset = n * H * D + h * D;
    const scalar_t* q_head = q + head_offset;
    const scalar_t* k_head = k + head_offset;
    const scalar_t* v_head = v + head_offset;
    scalar_t* out_head = out + head_offset;

    const scalar_t* q_row = q_head + l * N * H * D;   // stride to position l
    scalar_t* out_row = out_head + l * N * H * D;

    // compute scores and max
    scalar_t max_score = -1e30f;
    for (int64_t s = 0; s < L; ++s) {
        const scalar_t* k_row = k_head + s * N * H * D;
        scalar_t dot = 0;
        for (int64_t d = 0; d < D; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }
    scalar_t sum_exp = 0;
    for (int64_t s = 0; s < L; ++s) {
        const scalar_t* k_row = k_head + s * N * H * D;
        scalar_t dot = 0;
        for (int64_t d = 0; d < D; ++d) {
            dot += q_row[d] * k_row[d];
        }
        dot = (dot - max_score) * scale;
        dot = expf(dot);
        sum_exp += dot;
    }
    scalar_t inv_sum = scalar_t(1) / sum_exp;
    for (int64_t d = 0; d < D; ++d) {
        scalar_t sum = 0;
        for (int64_t s = 0; s < L; ++s) {
            const scalar_t* k_row = k_head + s * N * H * D;
            scalar_t dot = 0;
            for (int64_t e = 0; e < D; ++e) {
                dot += q_row[e] * k_row[e];
            }
            dot = (dot - max_score) * scale;
            scalar_t weight = expf(dot) * inv_sum;
            sum += weight * v_head[s * N * H * D + d];
        }
        out_row[d] = sum;
    }
}

//------------------------------------------------------------------------------
// Host launchers (extern "C" for binding)
//------------------------------------------------------------------------------

// Attention launcher
// Inputs:
//   query, key, value: float pointers, shape (L, N, E), can be non-contiguous; strides passed
//   in_proj_weight: (3E, E) contiguous
//   in_proj_bias: (3E) contiguous
//   out_proj_weight: (E, E) contiguous
//   out_proj_bias: (E) contiguous
//   L, N, E, H (num_heads): int64_t
//   workspace: allocated by binding, size should be enough for q,k,v,attn_out_buffer, etc.
// Output:
//   output: (L, N, E) contiguous, attention output
//   attn_weights: (N, H, L, L) or (L,N,H,L)? Contract says returns [batch, num_heads, sequence_length, sequence_length] but checkpoint shows [2048,16,16] which is likely (L,N,H) but confusing. We'll follow PyTorch default: when need_weights=True and average_attn_weights=True, returns (N, H, L, S) if batch_first=False? Actually PyTorch returns attn_output_weights shape (N, H, L, S) when batch_first=False, and (N, H, L, S) always? Documentation: If need_weights=True, returns attention weights averaged across heads of shape (L, S) when need_weights=True and average_attn_weights=True? Wait, in recent PyTorch (1.13+), if need_weights=True, it returns (N, H, L, S) or (N, L, S) depending on average_attn_weights. The checkpoint value shape [2048, 16, 16] suggests maybe (L, N, H)?? Actually (2048,16,16) doesn't match any standard. We'll assume the contract expectation is that attn_weights has shape (N, H, L, L) but we need to verify later. For our implementation we will return (N, H, L, L) if average_attn_weights=False, but if True average over heads -> (N, L, L). PyTorch default average_attn_weights=True, need_weights=True, so output weights shape (N, L, L) when batch_first=False (sequence first). That would be (16, 2048, 2048)? That is huge. But checkpoint shows [2048, 16, 16] which could be (L, N, H) with H=16? No H=8. Hmm, maybe the checkpoint is mistaken or uses a different MHA implementation. We'll stick to PyTorch's actual behavior for our binding's return; the contract may be adjusted after runtime verification. For now, we'll compute attn_weights of shape (N, H, L, L) and let binding decide if average. However, we cannot return a massive tensor unnecessarily. Since the model discards weights (uses _), we can return a dummy empty tensor or the actual weights but shape may be wrong. The goal is to match reference step_001 exactly for debug. The reference debug_forward yields (attn_output, attn_weights) and expects exact structure and shape. According to task_contract checkpoint step_001: value is tuple with first item shape [16, 2048, 256] and second item shape [2048, 16, 16]. Wait, first item [16,2048,256] is the same as input, not [2048,16,256]. That suggests that the MHA is being called with batch_first=True internally? But model.py does not set batch_first. Let's re-examine: In model.py, x = torch.randn(batch_size, sequence_length, embed_dim) gives [16,2048,256]. Then self.multihead_attn(x,x,x) with default batch_first=False expects input shape (L, N, E) = (16,2048,256), so L=16, N=2048, E=256. Output is also (16,2048,256). Then repeat_interleave dim=1 gives (16,6144,256). Checkpoint step_001 first item [16,2048,256] matches. So output shape is [16,2048,256]. Second item of step_001 is [2048,16,16]. That could be attn_weights shape? If average_attn_weights=True, output would be (N, L, L) = (2048,16,16) because N=2048, L=16. Yes! That matches checkpoint [2048,16,16]. So attn_weights shape is (N, L, L) when batch_first=False and average_attn_weights=True. So N=2048, L=16, L=16. That is correct. Therefore we need to return attn_weights of shape (N, L, L) = (2048,16,16). That makes sense. The contract's earlier note about [batch, num_heads, sequence_length, sequence_length] was incorrect in that context.

Thus, our custom_multihead_attention_forward must return:
- attn_output: shape (L, N, E) = (16, 2048, 256)
- attn_weights: shape (N, L, L) = (2048, 16, 16) (averaged over heads) because need_weights=True default and average_attn_weights=True.

We need to compute average attention weights over heads.

Implementation plan:
1. Allocate temporary device buffers for q, k, v, attn_output_buffer (before out_proj), attn_weights (N, L, L), and maybe head-specific weights if needed. But we can compute directly into attn_weights.
2. Compute q = linear(query, Wq, bq) with shape (L,N,D) where D = E. Actually head_dim = D = E/H? Wait, projection to embed_dim, then split heads. Usually Q, K, V are projected to embed_dim (E) then split into H heads of head_dim = E/H. So q,k,v shape (L,N,E). Then we view as (L,N,H,head_dim).
3. For each batch n and head h, compute scores = q_{n,h} @ k_{n,h}^T / sqrt(head_dim), softmax, then output_{n,h} = softmax @ v_{n,h}. Accumulate output into attn_output_buffer and also accumulate softmax weights into attn_weights_avg[n] (sum over heads) then divide by H.
4. After computing all heads, apply out_proj: output = linear(attn_output_buffer, out_proj_weight, out_proj_bias).

We can write a kernel that does per-head attention and averages weights. To avoid multiple passes over L for each head, we can write a more optimized kernel using shared memory for scores within a block covering a whole (n,h) slice if L is small (16). Since L=16, we can fit scores in shared memory easily. But L can be large in general? The contract sequence_length=2048 but L is actually 16 due to shape arrangement. However, the input shape is [16,2048,256], so L=16, N=2048. So L is small, N large. We'll optimize for L small.

We'll launch one block per (n,h) pair, with block size = L (16 threads maybe, but better use blockDim.x = L and each thread handles one query position). Each block computes all scores for its (n,h) using shared memory to store Q and K for the head. Since L=16, we can load Q and K into shared memory (size 2 * L * head_dim). head_dim = E/H = 256/8 = 32. So shared mem per block = 2*16*32*4 = 4096 bytes, fine. Then each thread (representing a query position) computes dot products with all keys using shared memory, finds max, computes exp, sums, then writes attention weights to attn_weights_avg (accumulate with atomicAdd? Since multiple heads contribute, we can use atomicAdd on float for each entry). Alternatively, to avoid atomics, we can have each block write to a temporary buffer for its head, and later a reduction kernel sums heads. But that would require extra memory. Simpler: use atomicAdd to accumulate into attn_weights tensor (initialized to zero). AtomicAdd on float is supported. Then after all heads done, launch a kernel to divide by H (or we can divide inside the block after accumulating all heads? impossible since blocks run concurrently). So we need a separate kernel to scale by 1/H.

Memory allocation: We'll require binding to provide workspace pointers. The binding will allocate q, k, v, attn_output, attn_weights. We can compute required sizes and allocate in binding, but our launcher can accept pointers to these buffers. We'll specify that launcher takes additional pointers: q_buffer, k_buffer, v_buffer, attn_output_buffer, attn_weights (already allocated by binding). The binding will allocate enough memory and pass pointers. That keeps .cu free of torch/ATen.

But the contract for the binding only includes the original parameters; it doesn't mention extra buffers. However, the binding implementation can allocate temporaries internally before calling the launcher. The launcher signature we define is for the binding to call, not the Python interface. We'll define a function:

void custom_multihead_attention_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* attn_weights,
    float* q_buffer, float* k_buffer, float* v_buffer, float* attn_output_buffer,
    int64_t L, int64_t N, int64_t E, int64_t H,
    int64_t query_row_stride, int64_t query_col_stride,
    int64_t key_row_stride, int64_t key_col_stride,
    int64_t value_row_stride, int64_t value_col_stride,
    cudaStream_t stream
)

But to keep things simpler, we can compute projections inside the same kernel? That would be complex. We'll separate into steps:
1. Compute q, k, v via linear_projection_kernel into provided buffers (contiguous).
2. Compute attention output and average weights using a fused kernel.
3. Compute out_proj.
4. Scale attn_weights by 1/H.

We can also combine out_proj into step 2? Not necessary.

For repeat_interleave: simple kernel that copies elements. Input x shape (L, N, E), dim=1 means repeat along N. Output shape (L, N*repeats, E). We can write kernel that each thread copies one element from input to output. Input may be non-contiguous, so use strides. Output is contiguous. Straightforward.

We'll implement all kernels in this file. Let's write code.
