// Complete custom CUDA implementation of MultiheadAttention
// with support for padding mask, batch_first, optional bias.
// All operations are performed in a single forward pass.

#include <cuda_runtime.h>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return; \
        } \
    } while (0)

// Helper: get value of padding mask at (batch, time)
static __device__ __forceinline__ bool get_mask_val(const bool* __restrict__ mask, int b, int t, int T) {
    return mask[b * T + t];
}

// Kernel: Compute QKV projections, attention, and output projection in one pass.
// Each thread block handles one (batch, head) pair, but we use T threads per block,
// each thread processes one query position. The K and V projections are computed
// in shared memory.
__global__ void multihead_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ in_proj_weight,
    const float* __restrict__ in_proj_bias,
    const float* __restrict__ out_proj_weight,
    const float* __restrict__ out_proj_bias,
    const bool* __restrict__ padding_mask,
    float* __restrict__ output,
    int B, int T, int E, int H,
    int batch_stride_q, int time_stride_q, int elem_stride_q,
    int batch_stride_k, int time_stride_k, int elem_stride_k,
    int batch_stride_v, int time_stride_v, int elem_stride_v,
    int batch_stride_out, int time_stride_out, int elem_stride_out,
    bool use_bias,
    bool mask_present
) {
    int head_dim = E / H;
    extern __shared__ float smem[];
    float* smem_keys = smem;
    float* smem_values = smem + T * head_dim;

    int batch = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    // Pointers to this thread's query, key, value input rows
    const float* q_input = query + batch * batch_stride_q + tid * time_stride_q;
    const float* k_input = key + batch * batch_stride_k + tid * time_stride_k;
    const float* v_input = value + batch * batch_stride_v + tid * time_stride_v;

    // Weight pointers for this head
    const float* q_weight = in_proj_weight + head * head_dim * E;
    const float* k_weight = in_proj_weight + E * E + head * head_dim * E;
    const float* v_weight = in_proj_weight + 2 * E * E + head * head_dim * E;

    const float* q_bias = (use_bias) ? (in_proj_bias + head * head_dim) : nullptr;
    const float* k_bias = (use_bias) ? (in_proj_bias + E + head * head_dim) : nullptr;
    const float* v_bias = (use_bias) ? (in_proj_bias + 2 * E + head * head_dim) : nullptr;

    // Compute K and V for this thread's position and store in shared memory
    for (int d = 0; d < head_dim; ++d) {
        float ksum = 0.0f;
        const float* w_row = k_weight + d * E;
        for (int e = 0; e < E; ++e) {
            ksum += w_row[e] * k_input[e * elem_stride_k];
        }
        if (use_bias) ksum += k_bias[d];
        smem_keys[tid * head_dim + d] = ksum;

        float vsum = 0.0f;
        w_row = v_weight + d * E;
        for (int e = 0; e < E; ++e) {
            vsum += w_row[e] * v_input[e * elem_stride_v];
        }
        if (use_bias) vsum += v_bias[d];
        smem_values[tid * head_dim + d] = vsum;
    }

    __syncthreads();

    // Compute Q for this thread
    // We assume head_dim <= 256 to fit in registers; otherwise dynamic allocation would be needed.
    float q_local[256];
    for (int d = 0; d < head_dim; ++d) {
        float qsum = 0.0f;
        const float* w_row = q_weight + d * E;
        for (int e = 0; e < E; ++e) {
            qsum += w_row[e] * q_input[e * elem_stride_q];
        }
        if (use_bias) qsum += q_bias[d];
        q_local[d] = qsum;
    }

    float scale = rsqrtf((float)head_dim);

    // Compute max for numerical stability
    float max_score = -FLT_MAX;
    for (int j = 0; j < T; ++j) {
        float score = 0.0f;
        const float* k_j = smem_keys + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q_local[d] * k_j[d];
        }
        score *= scale;
        if (mask_present && get_mask_val(padding_mask, batch, j, T)) {
            score = -INFINITY;
        }
        if (score > max_score) max_score = score;
    }

    // Compute exp sum and weighted sum
    float sum_exp = 0.0f;
    float acc[256];
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    for (int j = 0; j < T; ++j) {
        float score = 0.0f;
        const float* k_j = smem_keys + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q_local[d] * k_j[d];
        }
        score *= scale;
        if (mask_present && get_mask_val(padding_mask, batch, j, T)) {
            score = -INFINITY;
        }
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;
        const float* v_j = smem_values + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] += exp_score * v_j[d];
        }
    }

    // Write attention output for this query position into output for this head
    float* out_ptr = output + batch * batch_stride_out + tid * time_stride_out + head * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        out_ptr[d * elem_stride_out] = acc[d] / sum_exp;
    }
}

// Kernel: Apply output projection (matrix multiply by out_proj_weight and add bias) on the attention output.
// The input and output may point to the same memory (in-place projection).
__global__ void output_projection_kernel2(
    const float* input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int B, int T, int E,
    int input_batch_stride, int input_time_stride, int input_elem_stride,
    int output_batch_stride, int output_time_stride, int output_elem_stride,
    bool use_bias
) {
    // The attention result and final result share the same allocation.  A
    // conventional elementwise in-place GEMM is incorrect: one output column
    // can overwrite an input column still needed by another column.  Cache a
    // complete row before any thread writes to the output.
    extern __shared__ float row[];
    int row_id = blockIdx.x;
    int total_rows = B * T;
    if (row_id >= total_rows) return;
    int batch = row_id / T;
    int time = row_id - batch * T;
    const float* in_ptr = input + batch * input_batch_stride + time * input_time_stride;
    float* out_ptr = output + batch * output_batch_stride + time * output_time_stride;

    for (int e = threadIdx.x; e < E; e += blockDim.x)
        row[e] = in_ptr[e * input_elem_stride];
    __syncthreads();

    for (int col = threadIdx.x; col < E; col += blockDim.x) {
        float sum = 0.0f;
        const float* w = weight + col * E;
        for (int e = 0; e < E; ++e)
            sum += row[e] * w[e];
        if (use_bias) sum += bias[col];
        out_ptr[col * output_elem_stride] = sum;
    }
}

// Host launcher for multihead attention forward.
// Performs two kernels: 1) multihead_attention_kernel computes attention and writes
// concatenated heads into output; 2) output_projection_kernel2 applies out_proj and bias.
// Both kernels are launched on the provided stream.
// All pointer arguments are assumed to be device pointers to contiguous memory
// with the strides computed based on batch_first.
extern "C" void multihead_attention_forward_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* in_proj_weight,
    const float* in_proj_bias,
    const float* out_proj_weight,
    const float* out_proj_bias,
    const bool* padding_mask,
    float* output,
    int B, int T, int E, int H,
    bool batch_first,
    bool use_bias,
    bool mask_present,
    cudaStream_t stream
) {
    int batch_stride_q, time_stride_q, elem_stride_q;
    int batch_stride_k, time_stride_k, elem_stride_k;
    int batch_stride_v, time_stride_v, elem_stride_v;
    int batch_stride_out, time_stride_out, elem_stride_out;
    if (batch_first) {
        batch_stride_q = T * E; time_stride_q = E; elem_stride_q = 1;
        batch_stride_k = T * E; time_stride_k = E; elem_stride_k = 1;
        batch_stride_v = T * E; time_stride_v = E; elem_stride_v = 1;
        batch_stride_out = T * E; time_stride_out = E; elem_stride_out = 1;
    } else {
        batch_stride_q = E; time_stride_q = B * E; elem_stride_q = 1;
        batch_stride_k = E; time_stride_k = B * E; elem_stride_k = 1;
        batch_stride_v = E; time_stride_v = B * E; elem_stride_v = 1;
        batch_stride_out = E; time_stride_out = B * E; elem_stride_out = 1;
    }

    // Launch attention kernel: one block per (batch, head), T threads per block.
    // Shared memory size: K and V for all positions (T * head_dim each).
    dim3 grid1(B, H);
    dim3 block1(T); // T threads, each handles one query position
    size_t shared_mem_size = 2 * T * (E / H) * sizeof(float);
    multihead_attention_kernel<<<grid1, block1, shared_mem_size, stream>>>(
        query, key, value, in_proj_weight, in_proj_bias,
        out_proj_weight, out_proj_bias, padding_mask, output,
        B, T, E, H,
        batch_stride_q, time_stride_q, elem_stride_q,
        batch_stride_k, time_stride_k, elem_stride_k,
        batch_stride_v, time_stride_v, elem_stride_v,
        batch_stride_out, time_stride_out, elem_stride_out,
        use_bias, mask_present
    );

    // Launch output projection kernel: simple elementwise linear transformation.
    // It operates in-place on output.
    int threads = (E < 256) ? E : 256;
    int blocks = B * T;
    output_projection_kernel2<<<blocks, threads, static_cast<size_t>(E) * sizeof(float), stream>>>(
        output, out_proj_weight, out_proj_bias, output,
        B, T, E,
        batch_stride_out, time_stride_out, elem_stride_out,
        batch_stride_out, time_stride_out, elem_stride_out,
        use_bias
    );
}
