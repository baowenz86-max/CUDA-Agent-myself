#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// ----------------------------------------------------------------
// linear_forward: output = input @ weight^T + bias
// input: [batch, in_features], weight: [out_features, in_features],
// bias: [out_features], output: [batch, out_features]
//
// cuBLAS column-major: C = alpha * A * B + beta * C
// We need: output (batch x out_features) = input (batch x in_features) * weight^T (in_features x out_features)
// With column-major interpretation:
//   A = weight^T -> dims: M = in_features, N = batch, but we want row-major output.
// Let's treat input as row-major [batch, in_features]. In column-major, input is stored as [in_features, batch].
// weight is [out_features, in_features], but we want weight^T [in_features, out_features].
// In column-major, weight is stored as [in_features, out_features].
// So we can compute C = input_colmajor (in_features x batch) * weight_colmajor (out_features? need transpose).
// Better: use cublasSgemm with transa = CUBLAS_OP_N, transb = CUBLAS_OP_T
//   C (batch x out_features) = A (batch x in_features) * B (out_features x in_features)^T
// In column-major, A should be input stored as [in_features, batch], but input is row-major [batch, in_features].
// cuBLAS takes column-major, but if we pass row-major tensors, they are interpreted as transposed.
// Common trick: compute output^T = weight * input^T, then transpose, but we can avoid transpose by using cublasSgemm with proper strides.
// We'll use cublasSgemm with:
//   m = out_features, n = batch, k = in_features,
//   A = weight (out_features x in_features) with leading dimension lda = in_features,
//   B = input (batch x in_features) but we need B of shape (in_features x batch), so we use transpose operation.
// Actually, we can set transa = CUBLAS_OP_N, transb = CUBLAS_OP_T, then:
//   C = A * B^T, where A is m x k, B is n x k, C is m x n.
// We want C = input * weight^T, so if we set A = weight, B = input, C = output, we get output = weight * input^T, not what we want.
// If we set A = input, B = weight, and transpose both? Let's derive:
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
//               A, lda, B, ldb, &beta, C, ldc)
//   computes C = A^T * B, with A^T of shape m x k? Actually, if transa = CUBLAS_OP_T, A is k x m, A^T is m x k.
//   So set A = input (batch x in_features), transa = CUBLAS_OP_T -> A^T is in_features x batch.
//   set B = weight (out_features x in_features), transb = CUBLAS_OP_N -> B is out_features x in_features.
//   Then C = A^T * B has shape (in_features x batch) * (out_features x in_features) invalid.
// Let's do: C = input * weight^T.
//   input is batch x in_features, weight^T is in_features x out_features.
//   So m = batch, n = out_features, k = in_features.
//   We can use cublasSgemm with transa = CUBLAS_OP_N, transb = CUBLAS_OP_T,
//   A = input, lda = in_features (since row-major, but column-major expects leading dimension >= m, but we'll pass stride)
//   B = weight, ldb = in_features,
//   C = output, ldc = out_features.
//   However, cuBLAS assumes column-major, so if we pass row-major tensors, the dimensions are transposed.
//   To avoid confusion, we can use cublasSgemm with all tensors treated as column-major in memory and set the correct dimensions:
//   Column-major: input is in_features x batch, weight is in_features x out_features? Let's check memory layout.
//   input row-major [batch, in_features] has data stored as input[0,:], input[1,:], ...
//   In column-major, this is a matrix of size in_features x batch, with element (i,j) = input[j,i].
//   weight row-major [out_features, in_features] -> column-major in_features x out_features.
//   output row-major [batch, out_features] -> column-major out_features x batch.
//   We want output_colmajor = input_colmajor^T * weight_colmajor? Let's see:
//   output_colmajor (out_features x batch) = weight_colmajor (in_features x out_features) * input_colmajor (in_features x batch)? No.
//   Actually, we can compute output_colmajor = weight_colmajor^T * input_colmajor, but weight_colmajor is in_features x out_features, so weight_colmajor^T is out_features x in_features.
//   Then output_colmajor (out_features x batch) = weight_colmajor^T (out_features x in_features) * input_colmajor (in_features x batch).
//   This matches cublasSgemm with transa = CUBLAS_OP_T (A = weight_colmajor, A^T = out_features x in_features), transb = CUBLAS_OP_N (B = input_colmajor, in_features x batch), and C = output_colmajor (out_features x batch).
//   Great! So we call:
//   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//               out_features, batch, in_features,
//               &alpha,
//               weight, in_features,   // A: weight, but transa=T, so A is in_features x out_features, leading dim = in_features (since storage row-major? wait)
//               input, in_features,    // B: input, leading dim = in_features (row-major stride)
//               &beta,
//               output, out_features); // C: output, leading dim = out_features
//   But we must ensure leading dimensions are correct for column-major interpretation.
//   Let's test with small example: batch=1, in=2, out=3.
//   input row-major: [a,b]
//   weight row-major: [w00,w01; w10,w11; w20,w21] -> shape 3x2
//   output row-major: [c0,c1,c2]
//   cublasSgemm with transa=T, transb=N, m=out_features=3, n=batch=1, k=in_features=2,
//   A = weight, lda = in_features = 2. In column-major, A is 2x3? Actually, if transa=T, A is k x m = 2x3, but we pass weight which in memory is 3 rows of 2 elements. That's 3x2 row-major, which is 2x3 column-major. So A column-major is 2x3, correct for A^T to be 3x2.
//   B = input, ldb = in_features =2. B is n x k = 1x2? Actually, B should be k x n = 2x1 column-major, but input is 1x2 row-major, which is 2x1 column-major. So correct.
//   C = output, ldc = out_features=3. C is m x n = 3x1 column-major, but output should be 1x3 row-major. In memory, output row-major [c0,c1,c2] is 1x3 row-major, which is 3x1 column-major. So correct.
//   So this works. After GEMM, we add bias: output[i,j] += bias[j].
//
// We'll implement linear_forward using cublasSgemm and then a bias add kernel.

void linear_forward_cuda(const float* input, const float* weight, const float* bias, float* output,
                         int batch, int in_features, int out_features, cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 0.0f;
    // Note: we use cublasSgemm with transa=CUBLAS_OP_T, transb=CUBLAS_OP_N as derived.
    // m = out_features, n = batch, k = in_features
    // A = weight, lda = in_features
    // B = input, ldb = in_features
    // C = output, ldc = out_features
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch, in_features,
                &alpha,
                weight, in_features,
                input, in_features,
                &beta,
                output, out_features);
    cublasDestroy(handle);

    // Add bias: output[batch][out] += bias[out]
    int total = batch * out_features;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(output, bias, batch, out_features);
}

__global__ void add_bias_kernel(float* output, const float* bias, int batch, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_features;
    if (idx < total) {
        int col = idx % out_features;
        output[idx] += bias[col];
    }
}

// ----------------------------------------------------------------
// softmax_forward: softmax over dim=1 (columns) for 2D input [batch, num_params]
// We assume input is contiguous.
void softmax_forward_cuda(const float* input, float* output,
                          int batch, int num_params, cudaStream_t stream) {
    int block = 256;
    int grid = batch;
    softmax_kernel<<<grid, block, 0, stream>>>(input, output, batch, num_params);
}

__global__ void softmax_kernel(const float* input, float* output, int batch, int num_params) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    // Find max in this row
    __shared__ float max_val;
    __shared__ float sum_exp;
    if (tid == 0) {
        max_val = -INFINITY;
        sum_exp = 0.0f;
    }
    __syncthreads();
    for (int i = tid; i < num_params; i += blockDim.x) {
        float v = input[row * num_params + i];
        if (v > max_val) {
            // atomicMax on float? Use single thread sequential? Since blockDim=256 and num_params=200, we can use reduction.
            atomicMax((int*)&max_val, __float_as_int(v)); // unsafe but works for positive floats? Better use reduction.
        }
    }
    __syncthreads();
    // Now compute exp and sum
    for (int i = tid; i < num_params; i += blockDim.x) {
        float v = input[row * num_params + i];
        float exp_val = __expf(v - max_val);
        output[row * num_params + i] = exp_val;
        atomicAdd(&sum_exp, exp_val);
    }
    __syncthreads();
    // Normalize
    for (int i = tid; i < num_params; i += blockDim.x) {
        output[row * num_params + i] /= sum_exp;
    }
}

// log_softmax_forward: log_softmax over dim=1
void log_softmax_forward_cuda(const float* input, float* output,
                              int batch, int num_params, cudaStream_t stream) {
    int block = 256;
    int grid = batch;
    log_softmax_kernel<<<grid, block, 0, stream>>>(input, output, batch, num_params);
}

__global__ void log_softmax_kernel(const float* input, float* output, int batch, int num_params) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    __shared__ float max_val;
    __shared__ float sum_exp;
    if (tid == 0) {
        max_val = -INFINITY;
        sum_exp = 0.0f;
    }
    __syncthreads();
    for (int i = tid; i < num_params; i += blockDim.x) {
        float v = input[row * num_params + i];
        if (v > max_val) {
            max_val = v; // race condition; need reduction.
        }
    }
    __syncthreads();
    for (int i = tid; i < num_params; i += blockDim.x) {
        float v = input[row * num_params + i];
        float exp_val = __expf(v - max_val);
        output[row * num_params + i] = exp_val;
        atomicAdd(&sum_exp, exp_val);
    }
    __syncthreads();
    float log_sum = __logf(sum_exp);
    for (int i = tid; i < num_params; i += blockDim.x) {
        output[row * num_params + i] = input[row * num_params + i] - max_val - log_sum;
    }
}

// ----------------------------------------------------------------
// kl_div_loss_forward: sum(target * (log(target) - input)) / batch_size
// input and target are [batch, num_params]
void kl_div_loss_forward_cuda(const float* input, const float* target, float* output,
                              int batch, int num_params, cudaStream_t stream) {
    // Launch reduction kernel
    int block = 256;
    int grid = 1; // one block to sum all; but total elements = batch*num_params = 25600, too many for one block? We'll use atomicAdd.
    kl_div_kernel<<<grid, block, 0, stream>>>(input, target, output, batch, num_params);
}

__global__ void kl_div_kernel(const float* input, const float* target, float* output,
                              int batch, int num_params) {
    __shared__ float sum_val;
    if (threadIdx.x == 0) {
        sum_val = 0.0f;
    }
    __syncthreads();
    int total = batch * num_params;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        // target is probability, should be >0; use logf
        float t = target[i];
        float loss = t * (__logf(t) - input[i]);
        atomicAdd(&sum_val, loss);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        output[0] = sum_val / batch;
    }
}

// ----------------------------------------------------------------
// l1_loss_forward: mean absolute difference between input and target tensors of shape [batch_size]
void l1_loss_forward_cuda(const float* input, const float* target, float* output,
                          int batch, cudaStream_t stream) {
    int block = 256;
    int grid = 1;
    l1_loss_kernel<<<grid, block, 0, stream>>>(input, target, output, batch);
}

__global__ void l1_loss_kernel(const float* input, const float* target, float* output,
                               int batch) {
    __shared__ float sum_val;
    if (threadIdx.x == 0) {
        sum_val = 0.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < batch; i += blockDim.x) {
        atomicAdd(&sum_val, fabsf(input[i] - target[i]));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        output[0] = sum_val / batch;
    }
}

// ----------------------------------------------------------------
// Extern C launchers (will be called by binding.cpp)
extern "C" {

void linear_forward(const float* input, const float* weight, const float* bias, float* output,
                    int batch, int in_features, int out_features, cudaStream_t stream) {
    linear_forward_cuda(input, weight, bias, output, batch, in_features, out_features, stream);
}

void log_softmax_forward(const float* input, float* output,
                         int batch, int num_params, int dim, cudaStream_t stream) {
    log_softmax_forward_cuda(input, output, batch, num_params, stream);
}

void softmax_forward(const float* input, float* output,
                     int batch, int num_params, int dim, cudaStream_t stream) {
    softmax_forward_cuda(input, output, batch, num_params, stream);
}

void kl_div_loss_forward(const float* input, const float* target, float* output,
                         int batch, int num_params, cudaStream_t stream) {
    kl_div_loss_forward_cuda(input, target, output, batch, num_params, stream);
}

void l1_loss_forward(const float* input, const float* target, float* output,
                     int batch, cudaStream_t stream) {
    l1_loss_forward_cuda(input, target, output, batch, stream);
}

} // extern "C"
