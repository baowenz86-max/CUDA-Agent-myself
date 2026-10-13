#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      return err;                                                               \
    }                                                                           \
  } while (0)

// -----------------------------------------------------------------------------
// Layer normalization kernels
// -----------------------------------------------------------------------------

// Compute mean and variance for each row (last dimension is normalized)
// x: input tensor of shape (..., N), stride[last] = row_stride, stride[last] = 1 (or contiguous)
// We assume the last dimension is contiguous, but preceding dimensions may have arbitrary strides.
// grid: one block per row, blockDim.x threads handle elements of the row.
template <int BLOCK_SIZE>
__global__ void layer_norm_mean_var_kernel(const float* __restrict__ x,
                                           float* __restrict__ mean,
                                           float* __restrict__ var,
                                           int N,
                                           int rows,
                                           int64_t row_stride) {
    __shared__ float s_mean[BLOCK_SIZE];
    __shared__ float s_var[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * row_stride;
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Grid-stride loop over N elements of the row
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = row_ptr[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp-level reduction (optional, but simple block reduction with shared memory)
    s_mean[tid] = local_sum;
    s_var[tid] = local_sq_sum;
    __syncthreads();

    // Reduce within block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = s_mean[0] / N;
        float v = s_var[0] / N - m * m;
        mean[row] = m;
        var[row] = v;
    }
}

// Apply normalization and optional affine transformation
// mean and var arrays have one entry per row
// y = (x - mean) / sqrt(var + eps) * (weight ? weight[j] : 1) + (bias ? bias[j] : 0)
__global__ void layer_norm_apply_kernel(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ var,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float eps,
                                        int N,
                                        int rows,
                                        int64_t row_stride,
                                        int64_t out_row_stride,
                                        bool has_weight,
                                        bool has_bias) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * row_stride;
    float* y_row = y + row * out_row_stride;
    float m = mean[row];
    float v = var[row];
    float inv_std = rsqrtf(v + eps);

    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    // grid-stride loop over N
    for (int j = tid; j < N; j += total_threads) {
        float val = x_row[j];
        float norm = (val - m) * inv_std;
        if (has_weight && weight != nullptr) norm *= weight[j];
        if (has_bias && bias != nullptr) norm += bias[j];
        y_row[j] = norm;
    }
}

// -----------------------------------------------------------------------------
// nan_to_num kernel (default: nan -> 0, +-inf unchanged)
// -----------------------------------------------------------------------------
__global__ void nan_to_num_kernel(float* __restrict__ y,
                                  const float* __restrict__ x,
                                  int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = isnan(val) ? 0.0f : val;
    }
}

// -----------------------------------------------------------------------------
// add_bias kernel: scalar bias broadcast to all elements
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* __restrict__ y,
                                const float* __restrict__ x,
                                float bias,
                                int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + bias;
    }
}

// -----------------------------------------------------------------------------
// clamp kernel
// -----------------------------------------------------------------------------
__global__ void clamp_kernel(float* __restrict__ y,
                             const float* __restrict__ x,
                             float min_val,
                             float max_val,
                             int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}

// -----------------------------------------------------------------------------
// Launcher for layer_norm_default (no affine)
// -----------------------------------------------------------------------------
extern "C" cudaError_t layer_norm_default_launcher(
    const float* x,
    float* y,
    int N,               // normalized dimension (last dim size)
    int rows,            // number of rows = total_elements / N
    int64_t row_stride,  // stride between rows in elements (for x)
    int64_t out_row_stride, // stride between rows in output
    float eps,
    cudaStream_t stream) {

    // Allocate temporary mean and variance arrays on device
    float* d_mean;
    float* d_var;
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, rows * sizeof(float)));

    // Launch mean/var kernel
    const int block_size = 256;
    dim3 grid(rows);
    layer_norm_mean_var_kernel<block_size><<<grid, block_size, 0, stream>>>(
        x, d_mean, d_var, N, rows, row_stride);

    // Launch apply kernel
    const int apply_block = 256;
    dim3 apply_grid(rows);
    layer_norm_apply_kernel<<<apply_grid, apply_block, 0, stream>>>(
        x, y, d_mean, d_var, nullptr, nullptr, eps, N, rows, row_stride, out_row_stride, false, false);

    // Free temporary buffers after kernel completion (they are async, so we can't free immediately?)
    // Actually, cudaFree is stream-ordered, so it's safe to call after the kernels; they will be freed after execution.
    cudaFree(d_mean);
    cudaFree(d_var);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Launcher for layer_norm_affine (with weight and bias)
// -----------------------------------------------------------------------------
extern "C" cudaError_t layer_norm_affine_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int N,
    int rows,
    int64_t row_stride,
    int64_t out_row_stride,
    float eps,
    cudaStream_t stream) {

    float* d_mean;
    float* d_var;
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, rows * sizeof(float)));

    const int block_size = 256;
    dim3 grid(rows);
    layer_norm_mean_var_kernel<block_size><<<grid, block_size, 0, stream>>>(
        x, d_mean, d_var, N, rows, row_stride);

    const int apply_block = 256;
    dim3 apply_grid(rows);
    layer_norm_apply_kernel<<<apply_grid, apply_block, 0, stream>>>(
        x, y, d_mean, d_var, weight, bias, eps, N, rows, row_stride, out_row_stride, true, true);

    cudaFree(d_mean);
    cudaFree(d_var);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Launcher for nan_to_num
// -----------------------------------------------------------------------------
extern "C" cudaError_t nan_to_num_launcher(
    float* y,
    const float* x,
    int64_t n,
    cudaStream_t stream) {

    const int block = 256;
    const int64_t grid = (n + block - 1) / block;
    nan_to_num_kernel<<<grid, block, 0, stream>>>(y, x, n);
    return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Launcher for add_bias
// -----------------------------------------------------------------------------
extern "C" cudaError_t add_bias_launcher(
    float* y,
    const float* x,
    float bias,
    int64_t n,
    cudaStream_t stream) {

    const int block = 256;
    const int64_t grid = (n + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(y, x, bias, n);
    return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Launcher for clamp
// -----------------------------------------------------------------------------
extern "C" cudaError_t clamp_launcher(
    float* y,
    const float* x,
    float min_val,
    float max_val,
    int64_t n,
    cudaStream_t stream) {

    const int block = 256;
    const int64_t grid = (n + block - 1) / block;
    clamp_kernel<<<grid, block, 0, stream>>>(y, x, min_val, max_val, n);
    return cudaGetLastError();
}
