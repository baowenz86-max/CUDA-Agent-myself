#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute log_softmax along dim=0
// Input shape: rows x cols, reduction over rows (dim=0)
// Each block processes one column (or a group of columns)
__global__ void log_softmax_dim0_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    // Step 1: find max in column
    float max_val = -INFINITY;
    for (int r = 0; r < rows; ++r) {
        float v = input[r * cols + col];
        if (v > max_val) max_val = v;
    }

    // Step 2: compute sum of exp(x - max) using double for stability
    double sum = 0.0;
    for (int r = 0; r < rows; ++r) {
        float v = input[r * cols + col];
        sum += expf(v - max_val);
    }
    double log_sum = log(sum);

    // Step 3: write output = x - max - log_sum
    for (int r = 0; r < rows; ++r) {
        float v = input[r * cols + col];
        output[r * cols + col] = (float)(v - max_val - log_sum);
    }
}

// Kernel to concatenate two tensors along dim=0
// Output shape: rows_a + rows_b, cols (same cols)
__global__ void cat_dim0_kernel(const float* __restrict__ input_a,
                                 const float* __restrict__ input_b,
                                 float* __restrict__ output,
                                 int rows_a, int rows_b, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_a = rows_a * cols;
    int total_b = rows_b * cols;
    int total_out = total_a + total_b;
    if (idx >= total_out) return;

    if (idx < total_a) {
        output[idx] = input_a[idx];
    } else {
        output[idx] = input_b[idx - total_a];
    }
}

// Launcher for log_softmax along dim=0
extern "C" void log_softmax_dim_launcher(const float* input,
                                          float* output,
                                          int rows, int cols,
                                          cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;

    int block_size = 256;
    int grid_size = (cols + block_size - 1) / block_size;
    log_softmax_dim0_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, rows, cols);
}

// Launcher for cat along dim=0
extern "C" void cat_dim_launcher(const float* input_a,
                                  const float* input_b,
                                  float* output,
                                  int rows_a, int rows_b, int cols,
                                  cudaStream_t stream) {
    if (rows_a <= 0 || rows_b <= 0 || cols <= 0) return;

    int total_out = (rows_a + rows_b) * cols;
    int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    cat_dim0_kernel<<<grid_size, block_size, 0, stream>>>(
        input_a, input_b, output, rows_a, rows_b, cols);
}
