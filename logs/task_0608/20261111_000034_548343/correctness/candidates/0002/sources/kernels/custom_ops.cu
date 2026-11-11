#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declaration for kernel used in launcher
__global__ void equal_reduce_kernel_atomic(const float* lhs, const float* rhs, long long numel, int* global_result);

// Launcher for baddbmm_custom: out = input + mat1 @ mat2
// All tensors are float32, in row-major, with shapes [B, M, N], [B, M, K], [B, K, N]
// Supports arbitrary strides via leading dimensions for each matrix.
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
    // cuBLAS uses column-major, but we can treat row-major as column-major with transposed arguments.
    // For out = input + mat1 @ mat2, we compute using cublasSgemmStridedBatched.
    // In column-major terms: A = mat2^T (N x K), B = mat1^T (K x M), C = input^T (N x M)
    // but we want C = A * B + C, where C_out = input + mat1 @ mat2 => C_out^T = input^T + mat2^T @ mat1^T
    // Using cublasSgemmStridedBatched with opA = CUBLAS_OP_N for mat2^T? Let's set directly:
    // We'll use column-major interpretation: treat mat1 as MxK, mat2 as KxN, input as MxN.
    // However cuBLAS expects column-major storage (Fortran order), which is the transpose of row-major.
    // So if we pass pointers to row-major data, we can compute C = A * B where A is mat2^T (N x K), B is mat1^T (K x M),
    // and C is output^T (N x M). Then we set C = alpha * A * B + beta * C, with alpha=1, beta=1.
    // The strides for StridedBatched are given in elements between consecutive batches.
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    const float alpha = 1.0f;
    const float beta = 1.0f;
    
    // We have mat1: [B, M, K] (row-major) -> in column-major it's [B, K, M]? Actually, column-major of mat1 is transposed dimensions.
    // Common trick: compute out = input + mat1 @ mat2 using cuBLAS with column-major storage by treating
    // row-major as column-major and reversing dimensions.
    // We can compute C^T = B^T * A^T, where C = input + mat1 @ mat2.
    // Using cublasSgemmStridedBatched, we compute C' = alpha * A' * B' + beta * C',
    // where A' = mat2^T (N x K), B' = mat1^T (K x M), C' = input^T (N x M).
    // In column-major storage, A' (N x K) has leading dimension N, B' (K x M) leading dim K, C' (N x M) leading dim N.
    // The strides between batches for A' = N*K, for B' = K*M, for C' = N*M.
    // But our tensors are row-major, so mat2 row-major is [B, K, N], and its column-major view is [B, N, K] with leading dim N.
    // So we can directly pass pointers with appropriate leading dimensions and strides.
    
    // Let's use the following mapping:
    // A = mat2 (B x K x N) row-major -> column-major view B x N x K, so lda = N, strideA = K*N.
    // B = mat1 (B x M x K) row-major -> column-major view B x K x M, so ldb = K, strideB = M*K.
    // C = input (B x M x N) row-major -> column-major view B x N x M, so ldc = N, strideC = M*N.
    // Output will be written to output with same layout as input, i.e., column-major view B x N x M.
    // Operation: C = alpha * A * B + beta * C with A(NxK), B(KxM) produces NxM, which matches column-major C.
    // That corresponds to row-major output = input + mat1 @ mat2.
    
    // However, we must handle arbitrary strides? The contract says "supports strides", but for baddbmm, standard implem
    // may assume contiguous? We'll assume contiguous for simplicity and rely on binding to ensure contiguous tensors or
    // handle strides by using leading dimensions. We'll use the strides passed to support non-contiguous if needed.
    // For now, use the actual strides as leading dimensions and batch strides.
    
    int lda = static_cast<int>(mat2_stride_1); // N, because mat2 is [B,K,N] row-major => column-major leading dim N
    int ldb = static_cast<int>(mat1_stride_1); // K, because mat1 is [B,M,K] row-major => column-major leading dim K
    int ldc = static_cast<int>(input_stride_1); // N, because input is [B,M,N] row-major => column-major leading dim N
    
    // Batch strides in elements between matrices in the batch.
    // For column-major view, each batch stride = product of the two dimensions of the matrix.
    long long int strideA = mat2_stride_0 * mat2_stride_1; // K*N
    long long int strideB = mat1_stride_0 * mat1_stride_1; // M*K
    long long int strideC = input_stride_0 * input_stride_1; // M*N
    
    // cuBLAS expects long long for strided batched? Actually stride is long long in newer cuBLAS, int in older. We'll use long long.
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, // dimensions of column-major matrices: (N x K) * (K x M) -> (N x M)
        &alpha,
        mat2, lda, strideA,
        mat1, ldb, strideB,
        &beta,
        output, ldc, strideC,
        B
    );
    
    cublasDestroy(handle);
}

// Kernel for equal comparison: each thread block computes partial AND over its assigned elements.
__global__ void equal_reduce_kernel(const float* lhs, const float* rhs, long long numel, int* block_results) {
    extern __shared__ int sdata[]; // reduction in shared memory, but we can use warp shuffle.
    // We'll use a simple approach: each thread checks a strided set of elements and stores flag 1 if all equal, else 0.
    // Then reduce within block using shared memory and atomicAnd? Better: use block reduction and write one result per block.
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
    // Reduce within block using shared memory
    __shared__ int smem[256]; // max block size 256? We'll launch 256 threads per block.
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
    // Reduce within block using shared memory
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
    // We need to allocate device memory for block_results and final result.
    // We'll launch a kernel that reduces across the entire tensor.
    // For simplicity, we can do a two-pass reduction: first kernel reduces to one value per block,
    // then a second kernel reduces those block results to a single value.
    // Optionally, we can use atomicAnd on a global flag if all blocks agree, but that may be slower.
    // We'll do the two-pass approach.
    
    if (numel == 0) {
        // Empty tensors are considered equal if shapes match, but we assume shapes are checked by binding.
        // We'll return true for empty.
        return true;
    }
    
    const int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    // Limit number of blocks to avoid too many; but if numel is huge, grid-stride loop handles it.
    if (blocks > 65535) blocks = 65535; // max grid size for x dimension? We'll use grid stride, so any number works.
    // Actually we can use a grid-stride loop with a limited number of blocks, say 1024, and then reduce block results.
    // We'll set blocks = min(blocks, 1024) to keep block results small.
    if (blocks > 1024) blocks = 1024;
    
    // Allocate device memory for block results
    int* d_block_results;
    cudaMalloc(&d_block_results, blocks * sizeof(int));
    
    // Launch reduction kernel
    equal_reduce_kernel<<<blocks, threads, 0, stream>>>(lhs, rhs, numel, d_block_results);
    
    // Now reduce block results to a single int on device.
    // We can launch a second kernel with one block reducing these block results.
    // Or we can copy block results to host and check, but that would synchronize. We'll use a second kernel.
    int* d_final_result;
    cudaMalloc(&d_final_result, sizeof(int));
    // Launch kernel with one block of min(blocks, 1024) threads? But blocks may be > 1024? We limited to 1024, so one block of 1024 threads is enough.
    int reduce_threads = 1024;
    // Kernel to reduce block_results array of length blocks to one value.
    // We can reuse a simple reduction kernel.
    // We'll inline a small kernel here.
    // For simplicity, we can use thrust? Not allowed. So write a kernel.
    // We'll define a separate kernel or use a generic reduction.
    // But our current equal_reduce_kernel already reduces within a block. We can call a kernel that reduces an array of length blocks using one block.
    // We'll write another kernel with gridDim.x = 1, blockDim.x = 1024, and use shared memory reduction.
    // But block_results may be less than 1024, so we handle size < threads.
    
    // Define a second kernel:
    
    // Actually we can combine into one kernel using atomicAnd on a global flag, but that would be simpler.
    // We'll use a global int initialized to 1, and each block atomically ANDs its result. Then copy to host.
    // That avoids second kernel and extra allocation.
    // But atomicAnd on global memory is fine for bool. Let's use that for simplicity.
    
    // So modify plan:
    // - Allocate int d_result on device, initialize to 1 using cudaMemset.
    // - Launch equal_reduce_kernel that computes block result and then if block result is 0, atomically ANDs d_result with 0.
    // But we need to atomically update if partial result is 0, not always. We can do: if (smem[0] == 0) atomicAnd(d_result, 0);
    // But atomicAnd on global int is available.
    // - Then copy d_result to host and return as bool.
    // This is one kernel, one allocation.
    
    // Let's implement that.
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 1, sizeof(int));
    
    equal_reduce_kernel_atomic<<<blocks, threads, 0, stream>>>(lhs, rhs, numel, d_result);
    
    int h_result;
    cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // must synchronize to ensure copy completes
    
    cudaFree(d_block_results); // no longer used
    cudaFree(d_result);
    
    return h_result != 0;
}
