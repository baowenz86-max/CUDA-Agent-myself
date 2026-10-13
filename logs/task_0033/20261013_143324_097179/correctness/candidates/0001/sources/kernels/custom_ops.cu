// CUDA kernels for layer_norm and addcmul operations
#include <cuda_runtime.h>

// LayerNorm: normalizes over the last two dimensions (out_channels, length) per batch.
// Input x shape: [batch, out_channels, length], contiguous in memory.
// Weight and bias shape: [out_channels, length], contiguous in memory.
// eps is used for numerical stability.
// The mean and variance are computed per batch row of size = out_channels * length.
// Since size is large (204800), we use a two-pass approach:
// 1. Reduction to compute sum and sum of squares per row.
// 2. Normalize using the computed statistics.

// Kernel to compute partial sums per block for layer_norm reduction
__global__ void layer_norm_partial_sums_kernel(const float* __restrict__ x,
                                               float* __restrict__ partial_sums,
                                               float* __restrict__ partial_sq_sums,
                                               int row_size,
                                               int num_rows) {
    extern __shared__ float sdata[]; // shared memory for block reduction, size = 2 * blockDim.x
    float* s_sum = sdata;
    float* s_sq_sum = sdata + blockDim.x;

    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Initialize shared memory
    s_sum[tid] = 0.0f;
    s_sq_sum[tid] = 0.0f;

    const float* x_row = x + (size_t)row * row_size;

    // Grid-stride loop over row elements to compute sum and sum of squares
    for (int idx = blockIdx.y * block_size + tid; idx < row_size; idx += gridDim.y * block_size) {
        float val = x_row[idx];
        s_sum[tid] += val;
        s_sq_sum[tid] += val * val;
    }

    __syncthreads();

    // Reduction within block
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sq_sum[tid] += s_sq_sum[tid + offset];
        }
        __syncthreads();
    }

    // First thread writes block partial sums to global memory
    if (tid == 0) {
        int block_row = blockIdx.x;
        int block_col = blockIdx.y;
        int block_index = block_row * gridDim.y + block_col;
        partial_sums[block_index] = s_sum[0];
        partial_sq_sums[block_index] = s_sq_sum[0];
    }
}

// Kernel to compute final mean and variance per row from partial sums
__global__ void layer_norm_final_stats_kernel(const float* __restrict__ partial_sums,
                                              const float* __restrict__ partial_sq_sums,
                                              float* __restrict__ means,
                                              float* __restrict__ inv_stds,
                                              int row_size,
                                              int num_rows,
                                              float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    // Assume number of partial blocks per row is gridDim.y (passed via constant? need variable)
    // Instead, we pass num_partial_blocks as a function parameter, but since it's global, we'll use global variable? No,
    // Better: recompute or use a fixed gridDim.y from launch; but here we don't know. We'll pass it as an argument.
    // Since this is a second kernel, we need to pass num_partial_blocks.
    // Modify signature to include num_partial_blocks.
}

// The above final stats kernel is incomplete; we will instead avoid partial sums and use a simpler one-block-per-row approach
// because row_size is 204800, which is large but manageable with a single block using blockDim=256 and grid-stride within block.
// Actually, a single block can handle the whole row with grid-stride loop; then reduce within block.
// This avoids multiple kernel launches and global memory for partials.
// So we will implement a single kernel per row that computes mean and variance in shared memory and then normalizes.
// This is simpler and efficient enough for 204800 elements per row.
// We'll use one block per row, blockDim = 256, and each block does a grid-stride loop over the row to accumulate sum and sq_sum in shared memory.
// After reduction, compute mean and inv_std, then each thread normalizes its portion of the row.
// This is a classic fused layer_norm kernel.

__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ y,
                                  int row_size,
                                  int num_rows,
                                  float eps) {
    extern __shared__ float sdata[]; // size = 2 * blockDim.x
    float* s_sum = sdata;
    float* s_sq_sum = sdata + blockDim.x;

    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    s_sum[tid] = 0.0f;
    s_sq_sum[tid] = 0.0f;

    const float* x_row = x + (size_t)row * row_size;

    // Step 1: accumulate sum and sum of squares over row
    for (int idx = tid; idx < row_size; idx += block_size) {
        float val = x_row[idx];
        s_sum[tid] += val;
        s_sq_sum[tid] += val * val;
    }

    __syncthreads();

    // Reduce to thread 0
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sq_sum[tid] += s_sq_sum[tid + offset];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / row_size;
    float var = s_sq_sum[0] / row_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Step 2: normalize and apply weight/bias
    // weight and bias are 1D arrays of length row_size (same for all rows)
    for (int idx = tid; idx < row_size; idx += block_size) {
        float val = x_row[idx];
        float norm_val = (val - mean) * inv_std;
        y[row * (size_t)row_size + idx] = norm_val * weight[idx] + bias[idx];
    }
}

// Addcmul: output = input + value * tensor1 * tensor2
__global__ void addcmul_kernel(const float* __restrict__ input,
                               const float* __restrict__ tensor1,
                               const float* __restrict__ tensor2,
                               float* __restrict__ output,
                               int total_elements,
                               float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] + value * tensor1[idx] * tensor2[idx];
    }
}

// Launcher functions (extern "C" for binding)
extern "C" void layer_norm_launcher(const float* x,
                                    const float* weight,
                                    const float* bias,
                                    float* y,
                                    int batch,
                                    int out_channels,
                                    int length,
                                    float eps,
                                    cudaStream_t stream) {
    int row_size = out_channels * length;
    int num_rows = batch;
    int block_size = 256;
    dim3 grid(num_rows);
    size_t shared_mem = 2 * block_size * sizeof(float);
    layer_norm_kernel<<<grid, block_size, shared_mem, stream>>>(
        x, weight, bias, y, row_size, num_rows, eps);
}

extern "C" void addcmul_launcher(const float* input,
                                 const float* tensor1,
                                 const float* tensor2,
                                 float* output,
                                 int total_elements,
                                 float value,
                                 cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    addcmul_kernel<<<grid_size, block_size, 0, stream>>>(
        input, tensor1, tensor2, output, total_elements, value);
}
