// CUDA kernels for adaptive logsoftmax operations
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Simple linear: y = x @ W^T (no bias)
__global__ void linear_forward_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                      float* __restrict__ y, int batch, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * out_features) {
        int row = idx / out_features;
        int col = idx % out_features;
        const float* x_row = x + row * in_features;
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += x_row[k] * weight[col * in_features + k];
        }
        y[idx] = sum;
    }
}

// Absolute value elementwise
__global__ void absolute_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabsf(in[idx]);
}

// Compute head log softmax for all batch, but only first shortlist_size columns useful
__global__ void head_logsoftmax_kernel(const float* __restrict__ head_logits, float* __restrict__ head_log_probs,
                                       int batch, int head_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float* row = head_logits + b * head_size;
    float* out_row = head_log_probs + b * head_size;
    float max_val = -FLT_MAX;
    for (int i = 0; i < head_size; ++i) max_val = fmaxf(max_val, row[i]);
    float sum_exp = 0.0f;
    for (int i = 0; i < head_size; ++i) sum_exp += __expf(row[i] - max_val);
    float log_sum = logf(sum_exp);
    for (int i = 0; i < head_size; ++i) out_row[i] = row[i] - max_val - log_sum;
}

// Row-wise log softmax and add cluster logit
__global__ void row_wise_log_softmax_add_kernel(
    const float* __restrict__ input,       // [batch, ncols]
    float* __restrict__ output,             // [batch, ncols]
    const float* __restrict__ add_vals,     // [batch] values to add after logsoftmax
    int batch, int ncols)
{
    int b = blockIdx.x;
    if (b >= batch) return;
    const float* row = input + b * ncols;
    float* out_row = output + b * ncols;
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        sum_exp += __expf(row[i] - max_val);
    }
    smem[threadIdx.x] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum_exp = smem[0];
    float log_sum = logf(sum_exp);
    float add = add_vals[b];
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        out_row[i] = row[i] - max_val - log_sum + add;
    }
}

// Kernel to extract cluster logit from head_log_probs into a separate array for add_vals
__global__ void extract_column_kernel(const float* __restrict__ matrix, float* __restrict__ column,
                                      int batch, int total_cols, int col_index) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) column[b] = matrix[b * total_cols + col_index];
}

// Kernel for gathering target log probabilities and computing loss
__global__ void gather_and_loss_kernel(
    const float* __restrict__ log_probs,  // [batch, n_classes]
    const int64_t* __restrict__ target,   // [batch]
    float* __restrict__ output,           // [batch] (log probabilities of target)
    float* __restrict__ loss,             // scalar (mean of -output)
    int batch, int n_classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sum_nll[256];
    float local_nll = 0.0f;
    if (b < batch) {
        int64_t t = target[b];
        if (t < 0 || t >= n_classes) {
            output[b] = 0.0f; // fallback
        } else {
            output[b] = log_probs[b * n_classes + t];
        }
        local_nll = -output[b];
    }
    sum_nll[threadIdx.x] = local_nll;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_nll[threadIdx.x] += sum_nll[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(loss, sum_nll[0] / batch);
    }
}

extern "C" void adaptive_logsoftmax_logprob_launcher(
    const float* x,
    const float* head_weight,
    const float* tail0_w0,
    const float* tail0_w1,
    const float* tail1_w0,
    const float* tail1_w1,
    const int64_t* cutoffs,
    double div_value,
    float* output,
    int batch,
    int in_features,
    int head_size,
    int tail0_size0,
    int tail0_size1,
    int tail1_size0,
    int tail1_size1,
    int n_classes,
    int n_clusters,
    cudaStream_t stream)
{
    int threads = 256;
    // head logits
    float* head_logits;
    cudaMalloc(&head_logits, batch * head_size * sizeof(float));
    int blocks = (batch * head_size + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads, 0, stream>>>(x, head_weight, head_logits, batch, in_features, head_size);

    // head log probs
    float* head_log_probs;
    cudaMalloc(&head_log_probs, batch * head_size * sizeof(float));
    blocks = (batch + threads - 1) / threads;
    head_logsoftmax_kernel<<<blocks, threads, 0, stream>>>(head_logits, head_log_probs, batch, head_size);

    // copy head part to output, only shortlist_size = head_size - n_clusters columns
    int shortlist_size = head_size - n_clusters;
    cudaMemcpy2DAsync(output, n_classes * sizeof(float), head_log_probs, head_size * sizeof(float),
                      shortlist_size * sizeof(float), batch, cudaMemcpyDeviceToDevice, stream);

    // Tail 0
    float* hidden0;
    cudaMalloc(&hidden0, batch * tail0_size0 * sizeof(float));
    blocks = (batch * tail0_size0 + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads, 0, stream>>>(x, tail0_w0, hidden0, batch, in_features, tail0_size0);
    float* tail0_logits;
    cudaMalloc(&tail0_logits, batch * tail0_size1 * sizeof(float));
    blocks = (batch * tail0_size1 + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads, 0, stream>>>(hidden0, tail0_w1, tail0_logits, batch, tail0_size0, tail0_size1);

    float* cluster0_logits;
    cudaMalloc(&cluster0_logits, batch * sizeof(float));
    blocks = (batch + threads - 1) / threads;
    extract_column_kernel<<<blocks, threads, 0, stream>>>(head_log_probs, cluster0_logits, batch, head_size, shortlist_size + 0);

    float* tail0_log_probs;
    cudaMalloc(&tail0_log_probs, batch * tail0_size1 * sizeof(float));
    blocks = batch;
    row_wise_log_softmax_add_kernel<<<blocks, threads, 0, stream>>>(tail0_logits, tail0_log_probs, cluster0_logits, batch, tail0_size1);

    cudaMemcpy2DAsync(output + shortlist_size, n_classes * sizeof(float), tail0_log_probs, tail0_size1 * sizeof(float),
                      tail0_size1 * sizeof(float), batch, cudaMemcpyDeviceToDevice, stream);

    // Tail 1
    float* hidden1;
    cudaMalloc(&hidden1, batch * tail1_size0 * sizeof(float));
    blocks = (batch * tail1_size0 + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads, 0, stream>>>(x, tail1_w0, hidden1, batch, in_features, tail1_size0);
    float* tail1_logits;
    cudaMalloc(&tail1_logits, batch * tail1_size1 * sizeof(float));
    blocks = (batch * tail1_size1 + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads, 0, stream>>>(hidden1, tail1_w1, tail1_logits, batch, tail1_size0, tail1_size1);

    float* cluster1_logits;
    cudaMalloc(&cluster1_logits, batch * sizeof(float));
    blocks = (batch + threads - 1) / threads;
    extract_column_kernel<<<blocks, threads, 0, stream>>>(head_log_probs, cluster1_logits, batch, head_size, shortlist_size + 1);

    float* tail1_log_probs;
    cudaMalloc(&tail1_log_probs, batch * tail1_size1 * sizeof(float));
    blocks = batch;
    row_wise_log_softmax_add_kernel<<<blocks, threads, 0, stream>>>(tail1_logits, tail1_log_probs, cluster1_logits, batch, tail1_size1);

    cudaMemcpy2DAsync(output + shortlist_size + tail0_size1, n_classes * sizeof(float), tail1_log_probs, tail1_size1 * sizeof(float),
                      tail1_size1 * sizeof(float), batch, cudaMemcpyDeviceToDevice, stream);

    // free temporary buffers
    cudaFree(head_logits);
    cudaFree(head_log_probs);
    cudaFree(hidden0);
    cudaFree(tail0_logits);
    cudaFree(cluster0_logits);
    cudaFree(tail0_log_probs);
    cudaFree(hidden1);
    cudaFree(tail1_logits);
    cudaFree(cluster1_logits);
    cudaFree(tail1_log_probs);
}

extern "C" void absolute_launcher(const float* input, float* output, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    absolute_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

extern "C" void adaptive_logsoftmax_with_loss_launcher(
    const float* head_weight,
    const float* x,
    const int64_t* target,
    const float* tail0_w0,
    const float* tail0_w1,
    const float* tail1_w0,
    const float* tail1_w1,
    const int64_t* cutoffs,
    double div_value,
    float* output,
    float* loss,
    int batch,
    int in_features,
    int head_size,
    int tail0_size0,
    int tail0_size1,
    int tail1_size0,
    int tail1_size1,
    int n_classes,
    int n_clusters,
    cudaStream_t stream)
{
    float* log_probs;
    cudaMalloc(&log_probs, batch * n_classes * sizeof(float));
    adaptive_logsoftmax_logprob_launcher(
        x, head_weight, tail0_w0, tail0_w1, tail1_w0, tail1_w1,
        cutoffs, div_value, log_probs, batch, in_features,
        head_size, tail0_size0, tail0_size1, tail1_size0, tail1_size1,
        n_classes, n_clusters, stream);

    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    gather_and_loss_kernel<<<blocks, threads, 0, stream>>>(log_probs, target, output, loss, batch, n_classes);

    cudaFree(log_probs);
}
