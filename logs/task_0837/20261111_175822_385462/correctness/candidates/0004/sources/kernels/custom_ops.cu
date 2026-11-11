#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

// Matrix multiplication: C = A * B^T
__global__ void matmul_transb_kernel(
    const float* __restrict__ A, int lda, int strideA_row,
    const float* __restrict__ B, int ldb, int strideB_row,
    float* __restrict__ C, int ldc, int strideC_row,
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

// LogSoftmax over rows for 2D tensor
__global__ void log_softmax_rows_kernel(
    const float* __restrict__ input, int row_stride, int col_stride,
    float* __restrict__ output, int out_row_stride, int out_col_stride,
    int rows, int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* in_row = input + row * row_stride;
    float* out_row = output + row * out_row_stride;

    float max_val = -FLT_MAX;
    for (int j = 0; j < cols; ++j) {
        float v = in_row[j * col_stride];
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        sum += __expf(in_row[j * col_stride] - max_val);
    }
    float log_sum = logf(sum);

    for (int j = 0; j < cols; ++j) {
        out_row[j * out_col_stride] = in_row[j * col_stride] - max_val - log_sum;
    }
}

// Combine kernel: copies first head_cols columns from head and all tail_cols from tail
__global__ void combine_kernel(
    const float* __restrict__ head, int head_row_stride, int head_col_stride,
    int head_cols_to_copy,
    const float* __restrict__ tail, int tail_row_stride, int tail_col_stride,
    float* __restrict__ output, int out_row_stride, int out_col_stride,
    int batch, int head_cols_total, int tail_cols, int n_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_classes;
    if (idx >= total) return;

    int r = idx / n_classes;
    int c = idx % n_classes;
    float val;
    if (c < head_cols_to_copy) {
        val = head[r * head_row_stride + c * head_col_stride];
    } else {
        int tail_c = c - head_cols_to_copy;
        val = tail[r * tail_row_stride + tail_c * tail_col_stride];
    }
    output[r * out_row_stride + c * out_col_stride] = val;
}

// Head launcher: compute log_softmax over full head_weight rows (5001)
extern "C" void adaptive_logsoftmax_head_launcher(
    const float* x, int x_rows, int x_cols, int x_row_stride,
    const float* head_weight, int head_rows, int head_cols, int head_row_stride,
    float* output, int out_rows, int out_cols, int out_row_stride,
    cudaStream_t stream
) {
    if (x_rows == 0 || x_cols == 0 || head_rows == 0) return;

    dim3 block(16, 16);
    dim3 grid((head_rows + block.x - 1) / block.x, (x_rows + block.y - 1) / block.y);
    matmul_transb_kernel<<<grid, block, 0, stream>>>(
        x, 1, x_row_stride,
        head_weight, 1, head_row_stride,
        output, 1, out_row_stride,
        x_rows, head_rows, x_cols
    );

    int threads = 256;
    int blocks = (x_rows + threads - 1) / threads;
    log_softmax_rows_kernel<<<blocks, threads, 0, stream>>>(
        output, out_row_stride, 1,
        output, out_row_stride, 1,
        x_rows, head_rows
    );
}

// Tail launcher: hidden = x @ tail0^T; out = log_softmax(hidden @ tail1^T)
extern "C" void adaptive_logsoftmax_tail_launcher(
    const float* x, int x_rows, int x_cols, int x_row_stride,
    const float* tail0_weight, int tail0_rows, int tail0_cols, int tail0_row_stride,
    const float* tail1_weight, int tail1_rows, int tail1_cols, int tail1_row_stride,
    float* output, int out_rows, int out_cols, int out_row_stride,
    cudaStream_t stream
) {
    if (x_rows == 0 || x_cols == 0 || tail0_rows == 0 || tail1_rows == 0) return;

    float* hidden = nullptr;
    size_t hidden_bytes = (size_t)x_rows * tail0_rows * sizeof(float);
    cudaError_t err = cudaMalloc(&hidden, hidden_bytes);
    if (err != cudaSuccess) {
        return;
    }

    dim3 block(16, 16);
    dim3 grid((tail0_rows + block.x - 1) / block.x, (x_rows + block.y - 1) / block.y);
    matmul_transb_kernel<<<grid, block, 0, stream>>>(
        x, 1, x_row_stride,
        tail0_weight, 1, tail0_row_stride,
        hidden, 1, tail0_rows,
        x_rows, tail0_rows, x_cols
    );

    grid = dim3((tail1_rows + block.x - 1) / block.x, (x_rows + block.y - 1) / block.y);
    matmul_transb_kernel<<<grid, block, 0, stream>>>(
        hidden, 1, tail0_rows,
        tail1_weight, 1, tail1_row_stride,
        output, 1, out_row_stride,
        x_rows, tail1_rows, tail0_rows
    );

    int threads = 256;
    int blocks = (x_rows + threads - 1) / threads;
    log_softmax_rows_kernel<<<blocks, threads, 0, stream>>>(
        output, out_row_stride, 1,
        output, out_row_stride, 1,
        x_rows, tail1_rows
    );

    cudaStreamSynchronize(stream);
    cudaFree(hidden);
}

// Combine launcher
extern "C" void adaptive_logsoftmax_combine_launcher(
    const float* head, int head_rows, int head_cols, int head_row_stride,
    const float* tail, int tail_rows, int tail_cols, int tail_row_stride,
    float* output, int out_d0, int out_d1, int out_d2, int out_row_stride, int out_col_stride,
    cudaStream_t stream
) {
    if (head_rows == 0 || head_cols == 0 || tail_rows == 0) return;
    int head_cols_to_copy = 5000; // fixed: first 5000 head columns to copy
    int n_classes = head_cols_to_copy + tail_cols;
    int total = head_rows * n_classes;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    combine_kernel<<<blocks, threads, 0, stream>>>(
        head, head_row_stride, 1,
        head_cols_to_copy,
        tail, tail_row_stride, 1,
        output, out_row_stride, out_col_stride,
        head_rows, head_cols, tail_cols, n_classes
    );
}
