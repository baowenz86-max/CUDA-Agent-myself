// kernels/custom_ops.cu
// CUDA kernels for AdaptiveLogSoftmax custom ops.
// All launchers accept raw pointers, shape/stride info, and a CUDA stream.
// No PyTorch/ATen dependencies.

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Matrix multiplication: C = A * B^T
// A: [M, K] row-major with adim stride for inner dimension (typically 1)
// B: [N, K] row-major with bdim inner stride (typically 1)
// C: [M, N] row-major with cdim inner stride
// The kernel computes one element of C per thread using strided index calculation.
// It assumes all tensors are contiguous in the inner dimension (dim1 stride=1),
// but supports arbitrary leading (outer) strides. This covers the contract where
// all inputs are standard contiguous tensors, but we still guard out-of-bounds.
// -----------------------------------------------------------------------------
__global__ void matmul_transb_kernel(
    const float* __restrict__ A, int lda, int strideA_row, // A: M x K, lda usually 1, strideA_row = K if contiguous
    const float* __restrict__ B, int ldb, int strideB_row, // B: N x K
    float* __restrict__ C, int ldc, int strideC_row,      // C: M x N
    int M, int N, int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        const float* a_row = A + row * strideA_row;
        const float* b_row = B + col * strideB_row;
        for (int k = 0; k < K; ++k) {
            sum += a_row[k * lda] * b_row[k * ldb];
        }
        C[row * strideC_row + col * ldc] = sum;
    }
}

// -----------------------------------------------------------------------------
// LogSoftmax over rows for a 2D tensor.
// Input: [rows, cols] with stride (row_stride, col_stride usually 1)
// Output: same shape, log_softmax applied along dimension 1.
// -----------------------------------------------------------------------------
__global__ void log_softmax_rows_kernel(
    const float* __restrict__ input, int row_stride, int col_stride,
    float* __restrict__ output, int out_row_stride, int out_col_stride,
    int rows, int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in_row = input + row * row_stride;
    float* out_row = output + row * out_row_stride;

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < cols; ++j) {
        float v = in_row[j * col_stride];
        if (v > max_val) max_val = v;
    }

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        sum += __expf(in_row[j * col_stride] - max_val);
    }
    float log_sum = logf(sum);

    // Write log_softmax values
    for (int j = 0; j < cols; ++j) {
        out_row[j * out_col_stride] = in_row[j * col_stride] - max_val - log_sum;
    }
}

// -----------------------------------------------------------------------------
// Combine kernel: copies head and tail into a 3D tensor with shape [batch, 1, n_classes]
// head: [batch, head_cols]
// tail: [batch, tail_cols]
// output: [batch, 1, n_classes] with stride (n_classes, n_classes, 1) for contiguous
// This kernel writes directly into the unsqueezed layout.
// -----------------------------------------------------------------------------
__global__ void combine_kernel(
    const float* __restrict__ head, int head_row_stride, int head_col_stride,
    const float* __restrict__ tail, int tail_row_stride, int tail_col_stride,
    float* __restrict__ output, int out_row_stride, int out_col_stride, // output is 3D: [batch,1,nclasses]
    int batch, int head_cols, int tail_cols, int n_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_classes;
    if (idx >= total) return;

    int r = idx / n_classes;
    int c = idx % n_classes;
    float val;
    if (c < head_cols) {
        val = head[r * head_row_stride + c * head_col_stride];
    } else {
        int tail_c = c - head_cols;
        val = tail[r * tail_row_stride + tail_c * tail_col_stride];
    }
    // output has shape [batch, 1, n_classes], so row (batch) -> offset r*out_row_stride
    // column within last dim -> c*out_col_stride (out_col_stride should be 1)
    output[r * out_row_stride + c * out_col_stride] = val;
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C").
// These are called from binding.cpp. They perform no tensor checks; the binding
// layer must ensure inputs are on CUDA, float32, and valid shapes. We do handle
// zero sizes gracefully by returning early.
// -----------------------------------------------------------------------------

// Head log softmax: out = log_softmax(x @ head_weight^T)
// x: [batch, in_features]
// head_weight: [n_head_classes, in_features]
// output: [batch, n_head_classes]
// All tensors are assumed contiguous in the last dimension (stride[-1] = 1).
// We pass appropriate row strides.
extern "C" void adaptive_logsoftmax_head_launcher(
    const float* x, int x_rows, int x_cols, int x_row_stride,
    const float* head_weight, int head_rows, int head_cols, int head_row_stride,
    float* output, int out_rows, int out_cols, int out_row_stride,
    cudaStream_t stream
) {
    if (x_rows == 0 || x_cols == 0 || head_rows == 0) return;

    // Launch matrix multiplication: output = x * head_weight^T
    dim3 block(16, 16);
    dim3 grid((head_rows + block.x - 1) / block.x, (x_rows + block.y - 1) / block.y);
    matmul_transb_kernel<<<grid, block, 0, stream>>>(
        x, 1, x_row_stride, // A: x, lda=1 (contiguous cols)
        head_weight, 1, head_row_stride, // B: head_weight, lda=1
        output, 1, out_row_stride, // C: output
        x_rows, head_rows, x_cols
    );

    // Launch log_softmax on output rows
    int threads = 256;
    int blocks = (x_rows + threads - 1) / threads;
    log_softmax_rows_kernel<<<blocks, threads, 0, stream>>>(
        output, out_row_stride, 1,
        output, out_row_stride, 1,
        x_rows, head_rows
    );
}

// Tail log softmax: hidden = x @ tail0^T; out = log_softmax(hidden @ tail1^T)
// x: [batch, in_features]
// tail0: [hidden_size, in_features]
// tail1: [n_tail_classes, hidden_size]
// output: [batch, n_tail_classes]
extern "C" void adaptive_logsoftmax_tail_launcher(
    const float* x, int x_rows, int x_cols, int x_row_stride,
    const float* tail0_weight, int tail0_rows, int tail0_cols, int tail0_row_stride,
    const float* tail1_weight, int tail1_rows, int tail1_cols, int tail1_row_stride,
    float* output, int out_rows, int out_cols, int out_row_stride,
    cudaStream_t stream
) {
    if (x_rows == 0 || x_cols == 0 || tail0_rows == 0 || tail1_rows == 0) return;

    // Temporary buffer for hidden: [batch, hidden_size]
    // Since we only have raw pointers, we need to allocate scratch space. The binding
    // layer cannot provide workspace, so we allocate here using cudaMalloc.
    // For portability we use cudaMalloc and cudaFree. The overhead is acceptable for such small buffers.
    float* hidden = nullptr;
    size_t hidden_bytes = (size_t)x_rows * tail0_rows * sizeof(float);
    cudaError_t err = cudaMalloc(&hidden, hidden_bytes);
    if (err != cudaSuccess) {
        return;
    }

    // Step 1: hidden = x * tail0^T
    dim3 block(16, 16);
    dim3 grid((tail0_rows + block.x - 1) / block.x, (x_rows + block.y - 1) / block.y);
    matmul_transb_kernel<<<grid, block, 0, stream>>>(
        x, 1, x_row_stride,
        tail0_weight, 1, tail0_row_stride,
        hidden, 1, tail0_rows, // hidden row stride = hidden_size (contiguous)
        x_rows, tail0_rows, x_cols
    );

    // Step 2: output = hidden * tail1^T
    // hidden is [batch, hidden_size], tail1 is [n_tail_classes, hidden_size]
    grid = dim3((tail1_rows + block.x - 1) / block.x, (x_rows + block.y - 1) / block.y);
    matmul_transb_kernel<<<grid, block, 0, stream>>>(
        hidden, 1, tail0_rows, // hidden row stride = hidden_size
        tail1_weight, 1, tail1_row_stride,
        output, 1, out_row_stride,
        x_rows, tail1_rows, tail0_rows // K = hidden_size = tail0_rows
    );

    // Step 3: log_softmax on output rows
    int threads = 256;
    int blocks = (x_rows + threads - 1) / threads;
    log_softmax_rows_kernel<<<blocks, threads, 0, stream>>>(
        output, out_row_stride, 1,
        output, out_row_stride, 1,
        x_rows, tail1_rows
    );

    // Free hidden buffer, but must ensure all kernels have completed.
    // Since cudaFree after enqueuing may be unsafe, use cudaStreamSynchronize and then cudaFree.
    cudaStreamSynchronize(stream);
    cudaFree(hidden);
}

// Combine: output[b,0,c] = head[b,c] if c<5000 else tail[b,c-5000]
// head: [batch, head_cols]
// tail: [batch, tail_cols]
// output: [batch, 1, n_classes]
extern "C" void adaptive_logsoftmax_combine_launcher(
    const float* head, int head_rows, int head_cols, int head_row_stride,
    const float* tail, int tail_rows, int tail_cols, int tail_row_stride,
    float* output, int out_d0, int out_d1, int out_d2, int out_row_stride, int out_col_stride,
    cudaStream_t stream
) {
    if (head_rows == 0 || head_cols == 0 || tail_rows == 0) return;
    int n_classes = head_cols + tail_cols;
    int total = head_rows * n_classes;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    combine_kernel<<<blocks, threads, 0, stream>>>(
        head, head_row_stride, 1,
        tail, tail_row_stride, 1,
        output, out_row_stride, out_col_stride,
        head_rows, head_cols, tail_cols, n_classes
    );
}
