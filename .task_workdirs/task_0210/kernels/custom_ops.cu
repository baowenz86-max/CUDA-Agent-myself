// kernels/custom_ops.cu
// CUDA kernels and launchers for custom ops (no torch/ATen/c10/pybind)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Forward declaration of LSTM launcher (implemented in third-stage C++ file)
extern "C" void lstm_forward_launcher(
    float* output, float* hn, float* cn,
    const float* input,
    const float* weight_ih_l0, const float* weight_hh_l0,
    const float* bias_ih_l0, const float* bias_hh_l0,
    const float* weight_ih_l1, const float* weight_hh_l1,
    const float* bias_ih_l1, const float* bias_hh_l1,
    int batch, int seq_len, int input_size, int hidden_size,
    int num_layers, bool training, bool bidirectional, bool batch_first,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// nan_to_num kernel: replaces NaN with 0, +Inf with FLT_MAX, -Inf with -FLT_MAX
// -----------------------------------------------------------------------------
__global__ void nan_to_num_kernel(float* out, const float* in, long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        if (isnan(val)) {
            out[idx] = 0.0f;
        } else if (isinf(val)) {
            out[idx] = (val > 0) ? FLT_MAX : -FLT_MAX;
        } else {
            out[idx] = val;
        }
    }
}

// launcher for nan_to_num
extern "C" void nan_to_num_forward_launcher(float* output, const float* input, long long numel, cudaStream_t stream) {
    const int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    // Grid cap check: if grid_size > INT_MAX, use multiple wave loop, but numel could be large.
    // Here batch*seq*out_size = 512*500*10 = 2,560,000, grid_size ~ 10000 < 2^31-1, safe.
    nan_to_num_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(output, input, numel);
}

// -----------------------------------------------------------------------------
// matmul via cuBLAS: output = input @ bias
// input: [batch, seq_len, hidden_size] (contiguous)
// bias:  [hidden_size, output_size] (contiguous)
// output: [batch, seq_len, output_size]
// For each batch element, perform C = A * B where A is (seq_len, hidden_size), B is (hidden_size, output_size)
// Use cublasSgemm with row-major (CUBLAS_OP_N) after transposing mentally.
// Since cuBLAS is column-major by default, we can treat our row-major matrices as column-major with swapped dimensions and use CUBLAS_OP_T or N appropriately.
// Approach: Use cublasSgemm with:
//   m = output_size, n = seq_len, k = hidden_size
//   A = bias^T (hidden_size, output_size) -> use CUBLAS_OP_T on bias (output_size, hidden_size)
//   B = input^T (hidden_size, seq_len) -> use CUBLAS_OP_T on input (seq_len, hidden_size)
//   C = output^T (output_size, seq_len) -> result is column-major, which is row-major output (seq_len, output_size)
// Or simpler: use cublasGemmEx with CUBLAS_OP_N for row-major? cuBLAS does not have row-major but we can use cublasSgemm with transposes.
// Standard trick: C = A * B with row-major is equivalent to cublas column-major on transposes.
// We'll implement using cublasSgemm with CUBLAS_OP_T for both A and B, resulting in column-major C of shape (output_size, seq_len), which is exactly our desired row-major C (seq_len, output_size) if we interpret the same buffer appropriately.
// However, cublas output is column-major, but we want row-major. If m=output_size, n=seq_len, lda=hidden_size, ldb=hidden_size, ldc=output_size,
// then C column-major (output_size rows, seq_len columns) corresponds to row-major (seq_len rows, output_size columns). Yes.
// So call:
// cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
//             output_size, seq_len, hidden_size,
//             &alpha,
//             bias, hidden_size,  // A = bias^T (output_size x hidden_size), leading dim hidden_size
//             input, hidden_size, // B = input^T (hidden_size x seq_len), leading dim hidden_size
//             &beta,
//             output, output_size);
// For multiple batches, we could loop with pointer offsets. Or use cublasSgemmStridedBatched.
// For simplicity and correctness, loop over batch dimension, which may be acceptable but performance may be lower.
// Given batch=512, seq_len=500, hidden=200, output=10, total computation is 512*500*200*10 = 512 million MACs. cuBLAS per batch may have overhead but likely okay.
// We can later optimize with strided batched if needed. But for initial implementation, loop.
extern "C" void matmul_forward_launcher(float* output, const float* input, const float* bias,
                                      int batch, int seq_len, int hidden_size, int output_size,
                                      cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    for (int b = 0; b < batch; ++b) {
        const float* A = input + (long long)b * seq_len * hidden_size;
        float* C = output + (long long)b * seq_len * output_size;
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_T,
                    output_size, seq_len, hidden_size,
                    &alpha,
                    bias, hidden_size,           // A: bias^T (output_size x hidden_size)
                    A, hidden_size,               // B: input^T (hidden_size x seq_len)
                    &beta,
                    C, output_size);              // C: row-major seq_len x output_size
    }
    cublasDestroy(handle);
}
