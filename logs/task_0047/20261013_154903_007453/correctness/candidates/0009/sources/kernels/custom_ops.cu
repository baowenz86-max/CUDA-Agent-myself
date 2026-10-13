#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error at %s:%d status %d\n", __FILE__, __LINE__, (int)status); \
            return; \
        } \
    } while (0)

// Kernel: add bias to each row of output matrix
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float b = bias[col];
        for (int row = 0; row < rows; ++row) {
            output[row * cols + col] += b;
        }
    }
}

// Launcher: custom_linear
// input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]
// output: [batch, out_features]
extern "C" void custom_linear_launcher(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features,
    cudaStream_t stream) {
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Compute output = input * weight^T + bias
    // Using cuBLAS: C = alpha * op(A) * op(B) + beta * C
    // For row-major matrices, the standard approach is to treat them as column-major with swapped dimensions.
    // We compute output^T = weight * input^T, which is equivalent.
    // output is [batch, out] row-major, but viewed as column-major [out, batch].
    // weight is [out, in] row-major, viewed as column-major [in, out] which is weight^T.
    // input is [batch, in] row-major, viewed as column-major [in, batch] which is input^T.
    // So we call cublasSgemm with op(A)=CUBLAS_OP_N (weight^T), op(B)=CUBLAS_OP_N (input^T),
    // resulting in C_col = weight^T * input^T = (input * weight^T)^T = output^T.
    // Dimensions: m = out, n = batch, k = in.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            out_features, batch_size, in_features,
                            &alpha, weight, in_features, input, in_features,
                            &beta, output, out_features));
    
    if (bias != nullptr) {
        int block_size = 256;
        int grid_size = (out_features + block_size - 1) / block_size;
        add_bias_kernel<<<grid_size, block_size, 0, stream>>>(output, bias, batch_size, out_features);
        CUDA_CHECK(cudaGetLastError());
    }
    CUBLAS_CHECK(cublasDestroy(handle));
}

// Kernel: fused KLDiv loss per row
__global__ void kldiv_loss_kernel(const float* x, const float* target, float* output,
                                 int rows, int cols, float inv_rows) {
    __shared__ float sx[256];
    __shared__ float st[256];
    __shared__ float sdata[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    
    // Load row into shared memory, and compute max
    float max_x = -1e30f, max_t = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float xv = x[row * cols + i];
        float tv = target[row * cols + i];
        sx[i] = xv;
        st[i] = tv;
        max_x = fmaxf(max_x, xv);
        max_t = fmaxf(max_t, tv);
    }
    // Reduce max_x
    sdata[tid] = max_x;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    max_x = sdata[0];
    __syncthreads();
    // Reduce max_t
    sdata[tid] = max_t;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    max_t = sdata[0];
    __syncthreads();
    
    // Compute exp sums
    float sum_exp_x = 0.0f, sum_exp_t = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum_exp_x += expf(sx[i] - max_x);
        sum_exp_t += expf(st[i] - max_t);
    }
    sdata[tid] = sum_exp_x;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    sum_exp_x = sdata[0];
    __syncthreads();
    sdata[tid] = sum_exp_t;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    sum_exp_t = sdata[0];
    __syncthreads();
    
    // Compute KL contribution
    float log_sum_x = logf(sum_exp_x);
    float log_sum_t = logf(sum_exp_t);
    float kl_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float t_soft = expf(st[i] - max_t) / sum_exp_t;
        float x_log = sx[i] - max_x - log_sum_x;
        if (t_soft > 0.0f) {
            kl_sum += t_soft * (logf(t_soft) - x_log);
        }
    }
    // Block reduce kl_sum
    sdata[tid] = kl_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sdata[0] * inv_rows);
    }
}

// Launcher for kldiv fused kernel
extern "C" void custom_kldiv_loss_launcher(
    const float* x, const float* target, float* output,
    int batch_size, int num_cols, cudaStream_t stream) {
    
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    
    dim3 block(256);
    dim3 grid(batch_size);
    float inv_rows = 1.0f / batch_size;
    kldiv_loss_kernel<<<grid, block, 0, stream>>>(x, target, output, batch_size, num_cols, inv_rows);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel: l1_loss_sum single block reduction
__global__ void l1_loss_sum_kernel(const float* x, const float* params, float* output,
                                   int batch_size, int num_cols, float inv_batch_size) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int total = batch_size * num_cols;
    float sum = 0.0f;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int col = idx % num_cols;
        float diff = x[idx] - params[col];
        sum += fabsf(diff);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = sdata[0] * inv_batch_size;
    }
}

// Launcher for l1_loss_sum fused kernel
extern "C" void custom_l1_loss_sum_launcher(
    const float* x, const float* params, float* output,
    int batch_size, int num_cols, cudaStream_t stream) {
    
    dim3 block(256);
    dim3 grid(1);
    float inv_batch_size = 1.0f / batch_size;
    l1_loss_sum_kernel<<<grid, block, 0, stream>>>(x, params, output, batch_size, num_cols, inv_batch_size);
    CUDA_CHECK(cudaGetLastError());
}
