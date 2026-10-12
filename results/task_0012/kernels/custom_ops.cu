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
// Linear projection kernel
// -----------------------------------------------------------------------------
__global__ void linear_projection_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int rows,
    int in_features,
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
// Inputs q, k, v are assumed to be contiguous tensors of shape (seq_len, batch, embed_dim)
// where embed_dim = num_heads * head_dim.
// The memory layout is (token, batch, feature) with feature grouped by head.
// We compute one query per thread (batch_head_query).
// -----------------------------------------------------------------------------
__global__ void mha_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    float* __restrict__ attn_weights,
    int batch,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int64_t total_queries = (int64_t)batch * num_heads * seq_len;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t qid = idx; qid < total_queries; qid += stride) {
        int b = qid / (num_heads * seq_len);
        int rem = qid % (num_heads * seq_len);
        int h = rem / seq_len;
        int qi = rem % seq_len;

        float max_logit = -1e30f;
        for (int ki = 0; ki < seq_len; ++ki) {
            const float* q_vec = q + (((int64_t)qi * batch + b) * num_heads + h) * head_dim;
            const float* k_vec = k + (((int64_t)ki * batch + b) * num_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float logit = dot / sqrtf((float)head_dim);
            if (logit > max_logit) max_logit = logit;
        }

        float sum_exp = 0.0f;
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

        float* out_row = output + ((int64_t)qi * batch + b) * num_heads * head_dim;
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
            if (attn_weights != nullptr) {
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
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int seq_len,
    int batch,
    int embed_dim,
    cudaStream_t stream
) {
    int rows = seq_len * batch;
    int out_features = 3 * embed_dim;
    int64_t total_out = (int64_t)rows * out_features;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    linear_projection_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, rows, embed_dim, out_features
    );
}

void mha_forward_launcher(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* attn_weights,
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
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
