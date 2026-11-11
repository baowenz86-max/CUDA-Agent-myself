#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declaration for kernel used in launcher
__global__ void equal_reduce_kernel_atomic(const float* lhs, const float* rhs, long long numel, int* global_result);

// Launcher for baddbmm_custom: out = input + mat1 @ mat2
// All tensors are float32, in row-major, with shapes [B, M, N], [B, M, K], [B, K, N]
// Inputs are guaranteed contiguous by binding layer; strides are standard.
extern "C" void baddbmm_custom_launcher(
    float* output,
    const float* input,
    const float* mat1,
    const float* mat2,
    int B, int M, int N, int K,
    long long int input_stride_0, long long int input_stride_1,
    long long int mat1_stride_0, long long int mat1_stride_1,
    long long int mat2_stride_0, long long int mat2_stride_1,
    cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    const float alpha = 1.0f;
    const float beta = 1.0f;
    
    // cuBLAS column-major interpretation: compute C^T = B^T * A^T
    // A = mat2 (B x K x N) -> column-major view B x N x K, lda = N, strideA = K*N
    // B = mat1 (B x M x K) -> column-major view B x K x M, ldb = K, strideB = M*K
    // C = input (B x M x N) -> column-major view B x N x M, ldc = N, strideC = M*N
    int lda = N;
    int ldb = K;
    int ldc = N;
    
    long long int strideA = static_cast<long long int>(K) * N;
    long long int strideB = static_cast<long long int>(M) * K;
    long long int strideC = static_cast<long long int>(M) * N;
    
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        mat2, lda, strideA,
        mat1, ldb, strideB,
        &beta,
        output, ldc, strideC,
        B
    );
    
    cublasDestroy(handle);
}

// Kernel for equal comparison with atomic result
__global__ void equal_reduce_kernel(const float* lhs, const float* rhs, long long numel, int* block_results) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int all_equal = 1;
    for (long long i = idx; i < numel; i += stride) {
        if (lhs[i] != rhs[i]) {
            all_equal = 0;
            break;
        }
    }
    __shared__ int smem[256];
    smem[tid] = all_equal;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = smem[tid] & smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_results[blockIdx.x] = smem[0];
    }
}

// Alternative kernel using atomicAnd for block results
__global__ void equal_reduce_kernel_atomic(const float* lhs, const float* rhs, long long numel, int* global_result) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int all_equal = 1;
    for (long long i = idx; i < numel; i += stride) {
        if (lhs[i] != rhs[i]) {
            all_equal = 0;
            break;
        }
    }
    __shared__ int smem[256];
    smem[tid] = all_equal;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = smem[tid] & smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0 && smem[0] == 0) {
        atomicAnd(global_result, 0);
    }
}

// Launcher for equal_custom: returns bool (true if all elements equal).
extern "C" bool equal_custom_launcher(
    const float* lhs,
    const float* rhs,
    long long numel,
    cudaStream_t stream)
{
    if (numel == 0) {
        return true;
    }
    
    const int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 1, sizeof(int));
    
    equal_reduce_kernel_atomic<<<blocks, threads, 0, stream>>>(lhs, rhs, numel, d_result);
    
    int h_result;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaFree(d_result);
    
    return h_result != 0;
}
