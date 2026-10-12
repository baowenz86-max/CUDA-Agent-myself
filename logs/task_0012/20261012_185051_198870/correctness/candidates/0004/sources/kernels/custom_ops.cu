// custom_ops.cu
#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Elementwise fused ReLU + sqrt kernel
// -----------------------------------------------------------------------------
__global__ void relu_sqrt_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = sqrtf(fmaxf(x, 0.0f));
    }
}

// -----------------------------------------------------------------------------
// Elementwise SiLU kernel (x * sigmoid(x))
// -----------------------------------------------------------------------------
__global__ void silu_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float sig = 1.0f / (1.0f + expf(-x));
        output[i] = x * sig;
    }
}

// -----------------------------------------------------------------------------
// Linear projection: input (seq_len * batch, embed_dim) -> output (seq_len * batch, out_features)
// Weight: (out_features, embed_dim), Bias: (out_features)
// This kernel computes one output element per thread, iterating over embed_dim.
// Grid covers total output elements. Input and output are treated as 2D row-major.
// -----------------------------------------------------------------------------
__global__ void linear_projection_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int rows,        // seq_len * batch
    int in_features, // embed_dim
    int out_features
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = (int64_t)rows * out_features;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = out_idx; idx < total_out; idx += grid_stride) {
        int row = idx / out_features;
        int col = idx % out_features;
        const float* x_row = input + (int64_t)row * in_features;
        const float* w_row = weight + (int64_t)col * in_features;
        float sum = bias != nullptr ? bias[col] : 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += x_row[k] * w_row[k];
        }
        output[idx] = sum;
    }
}

// -----------------------------------------------------------------------------
// Multi-head attention kernel
// Inputs:
//   q, k, v: pointers to projected slices (already split or accessed via base pointer)
//   Actually we'll pass the full projected QKV buffer and compute offsets internally.
//   Instead, we'll have separate kernels for each phase: in_proj, attention, out_proj.
// This kernel computes attention outputs and optionally attention weights.
// It assigns one thread per (batch, head, query) and loops over keys.
// Grid dimension: logical size = batch * num_heads * seq_len, block 256.
// weights_output: if not nullptr, writes softmax probabilities to (batch, num_heads, seq_len, seq_len).
// -----------------------------------------------------------------------------
__global__ void mha_attention_kernel(
    const float* __restrict__ q,   // projected query: shape (seq_len, batch, num_heads, head_dim) or (seq_len*batch, num_heads*head_dim) we use flattened with indexing
    const float* __restrict__ k,   // projected key:   shape (seq_len, batch, num_heads, head_dim)
    const float* __restrict__ v,   // projected value: shape (seq_len, batch, num_heads, head_dim)
    float* __restrict__ output,    // output of attention before out_proj: shape (seq_len*batch, embed_dim)
    float* __restrict__ attn_weights, // if non-null, shape (batch, num_heads, seq_len, seq_len)
    int batch,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int64_t total_queries = (int64_t)batch * num_heads * seq_len;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t qid = idx; qid < total_queries; qid += stride) {
        // decode qid to (batch, head, query_token)
        int b = qid / (num_heads * seq_len);
        int rem = qid % (num_heads * seq_len);
        int h = rem / seq_len;
        int qi = rem % seq_len;

        // Compute logits for all keys, find max for numerical stability
        float max_logit = -1e30f;
        // We'll loop twice: once for max, once for exp and sum
        // For each key token ki
        for (int ki = 0; ki < seq_len; ++ki) {
            // compute dot product between query and key
            // q index: (qi, b, h) in memory layout: ((qi * batch + b) * num_heads + h) * head_dim
            // k index: ((ki * batch + b) * num_heads + h) * head_dim
            const float* q_vec = q + (((int64_t)qi * batch + b) * num_heads + h) * head_dim;
            const float* k_vec = k + (((int64_t)ki * batch + b) * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            // scale
            float logit = dot / sqrtf((float)head_dim);
            if (logit > max_logit) max_logit = logit;
        }

        float sum_exp = 0.0f;
        // Allocate temporary array for exponentials? Since seq_len=640, head_dim=32, we could use shared memory per block
        // But each thread handles a different query; storing all exponentials would be large.
        // We'll do three passes: max, sum, weighted sum. For weights output, we need to store exponentials or recompute.
        // Option: recompute exponentials in the third pass and write weights if needed.
        // To save memory and bandwidth, we'll not store exponentials but recompute for output.

        // Pass 2: compute sum of exp(logit - max)
        for (int ki = 0; ki < seq_len; ++ki) {
            const float* q_vec = q + (((int64_t)qi * batch + b) * num_heads + h) * head_dim;
            const float* k_vec = k + (((int64_t)ki * batch + b) * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float logit = dot / sqrtf((float)head_dim);
            sum_exp += expf(logit - max_logit);
        }

        // Pass 3: compute output and optionally write weights
        // output layout: (seq_len * batch, embed_dim). For token qi (query), the output row is qi * batch + b
        // The output feature dimension is embed_dim = num_heads * head_dim.
        // The contribution from head h goes to output row (qi * batch + b) and columns h*head_dim .. h*head_dim+head_dim-1.
        // For each key ki, we compute weight = exp(logit - max)/sum_exp, then accumulate into output features.
        float* out_row = output + ((int64_t)qi * batch + b) * num_heads * head_dim;

        // Initialize to zero for this head's slice? We'll accumulate over keys, but we need to ensure the output row for this head is zeroed initially.
        // Better to zero the output before accumulation. We can do that in a separate kernel or here before loop.
        // Since each thread handles only one (b,h,qi), it writes to a specific 32-element segment in the output row. That segment is only written by this thread, so we can zero it first.
        for (int d = 0; d < head_dim; ++d) {
            out_row[h * head_dim + d] = 0.0f;
        }

        for (int ki = 0; ki < seq_len; ++ki) {
            const float* q_vec = q + (((int64_t)qi * batch + b) * num_heads + h) * head_dim;
            const float* k_vec = k + (((int64_t)ki * batch + b) * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float logit = dot / sqrtf((float)head_dim);
            float weight = expf(logit - max_logit) / sum_exp;
            const float* v_vec = v + (((int64_t)ki * batch + b) * num_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_row[h * head_dim + d] += weight * v_vec[d];
            }
            // write attention weight if requested
            if (attn_weights != nullptr) {
                // weight location: ((b * num_heads + h) * seq_len + qi) * seq_len + ki
                attn_weights[(((int64_t)b * num_heads + h) * seq_len + qi) * seq_len + ki] = weight;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// -----------------------------------------------------------------------------
extern "C" {

void relu_sqrt_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    relu_sqrt_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

void silu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    silu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

void in_proj_forward_launcher(
    const float* input,       // (seq_len, batch, embed_dim) contiguous
    const float* weight,      // (3*embed_dim, embed_dim)
    const float* bias,        // (3*embed_dim)
    float* output,            // (seq_len, batch, 3*embed_dim) contiguous
    int seq_len,
    int batch,
    int embed_dim,
    cudaStream_t stream
) {
    int rows = seq_len * batch; // number of rows for linear projection (each token/batch)
    int out_features = 3 * embed_dim;
    int64_t total_out = (int64_t)rows * out_features;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    // input is contiguous (seq_len, batch, embed_dim) but linear projection treats as (rows, in_features)
    // where rows = seq_len * batch, row index = token_seq * batch + batch?? Actually for (seq_len, batch, embed_dim),
    // the contiguous layout has stride: batch*embed_dim for seq, embed_dim for batch, 1 for feature.
    // To treat as (rows, in_features) with row-major, we need rows = seq_len * batch, and row index = token * batch + b.
    // But our input pointer is the original tensor; when we access input[row * in_features + k], that assumes row-major flattening of (seq_len, batch, embed_dim), which is not the case.
    // We need to be careful: the actual memory order is seq major, then batch, then feature. So the correct 2D indexing is (token * batch + b) for row, and k for column.
    // So row = token * batch + b. This is what we use in linear_projection_kernel by setting row = idx / out_features, but we need to map that row back to the original memory offset: row_original = token * batch + b, and input pointer offset = token * (batch * embed_dim) + b * embed_dim + k.
    // That is not equal to row * embed_dim + k because row = token * batch + b, and token*(batch*embed_dim)+b*embed_dim = (token*batch+b)*embed_dim = row*embed_dim. So actually it does equal row*embed_dim! Because (token * batch + b) * embed_dim = token * batch * embed_dim + b * embed_dim. Yes, that's correct. So input can be treated as (rows, embed_dim) with row index = token * batch + b. Great.
    linear_projection_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, rows, embed_dim, out_features
    );
}

void mha_forward_launcher(
    const float* q,           // projected query: (seq_len, batch, 3*embed_dim)?? Actually we need to split into heads.
    // We'll receive the combined projected QKV buffer: output of in_proj (seq_len, batch, 3*embed_dim)
    // We'll pass three pointers: q_ptr, k_ptr, v_ptr which point to the slices within that buffer.
    // The binding layer will pass these pointers separately.
    const float* k,
    const float* v,
    float* output,            // attention output before out_proj: (seq_len, batch, embed_dim) -> flattened (seq_len*batch, embed_dim)
    float* attn_weights,      // if non-null, (batch, num_heads, seq_len, seq_len)
    int batch,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    int64_t total_queries = (int64_t)batch * num_heads * seq_len;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    mha_attention_kernel<<<blocks, threads, 0, stream>>>(
        q, k, v, output, attn_weights,
        batch, seq_len, num_heads, head_dim
    );
}

void out_proj_forward_launcher(
    const float* input,       // attention output: (seq_len, batch, embed_dim) contiguous -> rows = seq_len*batch, in_features=embed_dim
    const float* weight,      // (embed_dim, embed_dim)
    const float* bias,        // (embed_dim)
    float* output,            // (seq_len, batch, embed_dim) contiguous
    int seq_len,
    int batch,
    int embed_dim,
    cudaStream_t stream
) {
    int rows = seq_len * batch;
    int out_features = embed_dim;
    int64_t total_out = (int64_t)rows * out_features;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    linear_projection_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, rows, embed_dim, out_features
    );
}

} // extern "C"
