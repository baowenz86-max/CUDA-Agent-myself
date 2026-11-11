// CUDA implementation of Multihead Attention forward
// This file contains CUDA kernels and launchers for the mha_forward operator.
// Dependencies: CUDA runtime and cuBLAS (for GEMM).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Define constants for clarity
constexpr int WARP_SIZE = 32;
constexpr float INV_SQRT_DIM = 1.0f / sqrtf(128.0f); // 1/sqrt(head_dim)

// Softmax kernel: input [N, L, L] (N = batch*heads), output same shape
__global__ void softmax_kernel(float* out, const float* in, int N, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * L;
    if (idx < total) {
        int n = idx / L;
        int i = idx % L;
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < L; ++j) {
            float v = in[n * L * L + i * L + j];
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < L; ++j) {
            float v = expf(in[n * L * L + i * L + j] - max_val);
            out[n * L * L + i * L + j] = v;
            sum += v;
        }
        for (int j = 0; j < L; ++j) {
            out[n * L * L + i * L + j] /= sum;
        }
    }
}

// Attention output kernel: out = softmax(Q*K^T) * V
// Inputs: Q [N, L, D], K [N, L, D], V [N, L, D], softmax_scores [N, L, L]
// Output: attn_out [N, L, D]
__global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                 const float* softmax_scores, float* attn_out,
                                 int N, int L, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * L * D) {
        int n = idx / (L * D);
        int remainder = idx % (L * D);
        int i = remainder / D;
        int d = remainder % D;
        float sum = 0.0f;
        for (int j = 0; j < L; ++j) {
            sum += softmax_scores[n * L * L + i * L + j] * V[n * L * D + j * D + d];
        }
        attn_out[idx] = sum;
    }
}

// Kernel to compute Q*K^T scaled by 1/sqrt(D) and add bias? no bias in MHA forward.
// Instead we use cuBLAS for GEMM operations, so no custom kernel for QK^T needed.
// But contract says no torch ops, we can use cuBLAS.

// Launcher function for mha_forward.
// Input tensors are assumed to be on GPU, contiguous for weights, x may be non-contiguous but we can assume contiguous for simplicity
// because binding layer can make it contiguous if needed.
// Shapes: x: [batch, seq_len, embed_dim] = [256,32,512]
// in_proj_weight: [3*embed_dim, embed_dim] = [1536,512]
// in_proj_bias: [1536]
// out_proj_weight: [embed_dim, embed_dim] = [512,512]
// out_proj_bias: [512]
// Outputs: attn_output [batch, seq_len, embed_dim] = [256,32,512]
//          attn_weights_avg [seq_len, batch, batch]? Actually contract says [32,256,256] (batch averaged over heads)
// We need to allocate these in binding and pass pointers here.

extern "C" void mha_forward_launcher(
    const float* x,             // [B, L, E]
    const float* in_proj_weight,// [3E, E]
    const float* in_proj_bias,  // [3E]
    const float* out_proj_weight,// [E, E]
    const float* out_proj_bias, // [E]
    float* attn_output,         // [B, L, E]  (preallocated by binding)
    float* attn_weights_avg,    // [L, B, B]? Actually shape [32,256,256] = [seq_len, batch*heads?, batch?] 
                                // Contract: attn_weights shape [32,256,256] (batch averaged). So it's [L, B, B]? 
                                // Wait batch=256? No, batch dimension is 256? Actually input shape [256,32,512], so batch=256, seq_len=32.
                                // Contract says attn_weights shape [32,256,256] which is [seq_len, batch, batch]. Yes.
    int B, int L, int E, int H, // H = num_heads = 4
    cudaStream_t stream)
{
    // Dimensions
    int N = B * H;          // total number of heads across batch
    int D = E / H;          // head dimension = 512/4 = 128

    // Workspace sizes
    // q, k, v: [B, L, E] each, or we can store as [N, L, D] for attention.
    // We'll allocate temporary buffers:
    // in_proj_output [B, L, 3E] (or separate q,k,v)
    // q [N, L, D], k [N, L, D], v [N, L, D]
    // scores [N, L, L], softmax_scores [N, L, L]
    // attn_out [N, L, D] = attention output per head
    // attn_out_reshaped [B, L, E] = concatenated heads
    // final output [B, L, E] after out_projection (already provided)

    // We'll use cuBLAS for GEMMs.
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Step 1: in_projection: proj = x @ in_proj_weight^T + in_proj_bias
    // x: [B*L, E] (if row-major contiguous, B*L rows, E cols)
    // in_proj_weight: [3E, E], need transpose to [E, 3E] because x @ W^T
    // Actually GEMM: C = A * B, where A is [M,K], B is [K,N].
    // For x @ in_proj_weight^T: A = x [B*L, E], B = in_proj_weight^T [E, 3E] -> C [B*L, 3E]
    // We need to compute in_proj_weight^T. We can use cublasSgemm with transa=CUBLAS_OP_N and transb=CUBLAS_OP_T.
    // cublasSgemm: C = alpha * op(A) * op(B) + beta * C,
    // op(A) is either A or A^T, op(B) similarly.
    // We want A = x (no transpose), B = in_proj_weight (transpose) -> op(B) = B^T.
    // So: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc)
    // M = B*L, N = 3E, K = E, lda = E, ldb = E, ldc = 3E
    // Need to allocate in_proj_output [B*L, 3E] = [B*L*3E] floats.

    // Allocate device memory for all temporaries using cudaMalloc.
    float *in_proj_output, *q, *k, *v, *scores, *softmax_scores, *attn_out, *attn_out_reshaped;
    cudaMalloc(&in_proj_output, (size_t)B * L * 3 * E * sizeof(float));
    cudaMalloc(&q, (size_t)N * L * D * sizeof(float));
    cudaMalloc(&k, (size_t)N * L * D * sizeof(float));
    cudaMalloc(&v, (size_t)N * L * D * sizeof(float));
    cudaMalloc(&scores, (size_t)N * L * L * sizeof(float));
    cudaMalloc(&softmax_scores, (size_t)N * L * L * sizeof(float));
    cudaMalloc(&attn_out, (size_t)N * L * D * sizeof(float));
    cudaMalloc(&attn_out_reshaped, (size_t)B * L * E * sizeof(float));

    // Perform GEMM for in_projection
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                B * L, 3 * E, E,
                &alpha,
                x, E,            // A row-major [B*L, E], leading dimension E
                in_proj_weight, E, // B [3E, E], but we pass B^T with leading dim E (since B is row-major [3E, E], B^T is [E,3E] with ld E? Actually for CUBLAS_OP_T, B is stored column-major, but we have row-major. 
                // Wait careful: cuBLAS uses column-major layout by default. PyTorch tensors are row-major and contiguous, but cuBLAS can interpret them as column-major if we transpose operations.
                // For a row-major matrix A of dimensions M x K, cuBLAS with CUBLAS_OP_N treats it as column-major with dimensions K x M (since column-major storage means A_col[j + i*lda] = A_row[i*K + j]).
                // So to perform C = x @ in_proj_weight^T in row-major, we can use cuBLAS in column-major by noting:
                // Column-major interpretation: x_col = x row-major transposed? Let's think carefully.
                // Simpler: we can convert x to column-major? But copying would be expensive.
                // Standard trick: for row-major A (MxK), B (KxN), C (MxN), we call cublasSgemm with:
                // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                //            N, M, K, &alpha, B, N, A, K, &beta, C, N)
                // i.e., swap A and B and use N for dimensions? But M,N,K correspond to columns of op(A) and op(B) in column-major.
                // Actually for row-major, we often compute C = A * B by calling cublasSgemm with op(A)=CUBLAS_OP_T, op(B)=CUBLAS_OP_T, and dimensions N, M, K.
                // Let's derive: We want C_row (MxN) = A_row (MxK) * B_row (KxN).
                // In column-major, A_col = A_row^T (KxM), B_col = B_row^T (NxK), C_col = C_row^T (NxM).
                // So C_col = B_col * A_col (since (AB)^T = B^T A^T). So we can compute C_col = B_col * A_col using column-major GEMM with op(A)=B_col (no transpose), op(B)=A_col (no transpose). 
                // This corresponds to cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_col, ldb, A_col, lda, &beta, C_col, ldc)
                // But B_col is row-major B (NxK) stored as column-major, so its leading dimension is K (since B_col has N rows, K cols? Wait B_col = B^T, so dimensions N x K, stored column-major means leading dimension N? Let's be precise.
                // For a row-major tensor of shape [M,N], its data pointer can be interpreted as column-major with shape [N,M] and leading dimension N (since element (i,j) row-major is at i*N+j, column-major with shape N x M has element (j,i) at j*M+i? Not matching).
                // This is getting confusing. The common approach: use cublasSgemm with CUBLAS_OP_T for both A and B to compute standard row-major multiplication. That is, for row-major A (MxK) and B (KxN), call:
                // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, K, &beta, C, N)
                // where M,N,K are the original row-major dimensions, but the order of arguments changes: the first two dimensions are N and M, and the K is same.
                // Then A and B pointers are the row-major tensors, with leading dimensions N and K respectively.
                // Let's verify with small example: A (2x3), B (3x4), C=AB (2x4). Call cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 4, 2, 3, &alpha, B, 4, A, 3, &beta, C, 4). This works because op(B) treats B as col-major? Actually many examples use this trick.
                // We'll use that trick.

                // For in_projection: A = x (M=B*L, K=E), B = in_proj_weight (K=3E?, N=???)
                // Actually in_proj_weight is [3E, E], so we want C = x @ in_proj_weight^T => dimensions: x [B*L, E], in_proj_weight^T [E, 3E] => C [B*L, 3E].
                // So row-major A: M=B*L, K=E; B: K=E, N=3E.
                // Use trick: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, K, &beta, C, N)
                // Here B pointer is in_proj_weight, leading dimension N (which is 3E). A pointer is x, leading dimension K (which is E). C pointer is in_proj_output, leading dimension N (3E).
                // Dimensions passed: N=3E, M=B*L, K=E.
                // Let's check: op(B) with CUBLAS_OP_T means B is transposed; B passed is row-major [E,3E]? Wait original in_proj_weight is [3E,E], so in_proj_weight^T is [E,3E]. In row-major, in_proj_weight^T has E rows and 3E cols. The trick expects B as row-major [K,N] with leading dimension N? Actually for the trick, B should be the row-major matrix of shape [K,N]. When we pass B with CUBLAS_OP_T, it treats it as column-major of shape [N,K], and op(B) gives [K,N]? Hmm.
                // Better to refer to known implementation: https://stackoverflow.com/questions/14595750/transpose-matrix-multiplication-in-cublas-how-to
                // The standard is: for row-major matrices, use cublasSgemm with CUBLAS_OP_N for both and swap A and B, and use appropriate leading dimensions.
                // Alternative: We can simply allocate transposed weights using cudaMemcpy or cublasSgeam to avoid confusion. But performance would suffer.
                // Given complexity, we can use the known formula: To compute C = A * B for row-major A (MxK), B (KxN), call cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, K, &beta, C, N). Yes, this is correct.

                // So for in_projection:
                int M = B * L;
                int K1 = E;
                int N1 = 3 * E;
                // B matrix is in_proj_weight^T? Actually we pass in_proj_weight as B in the call, but we want B to be in_proj_weight^T, which has shape [E,3E]. in_proj_weight is [3E,E]. So the trick expects B to be [K,N] row-major. But we have in_proj_weight [N,K] row-major. That's the transpose. So we need to pass in_proj_weight as A? Let's adjust.
                // We need to compute C = x * W^T. x is [M,K], W is [N,K]. Then W^T is [K,N]. So B should be W^T [K,N]. But we have W [N,K], so we can pass W as B with CUBLAS_OP_T to treat it as W^T. In the trick, if we pass B = W (row-major [N,K]) with CUBLAS_OP_T, the dimensions for cublasSgemm must be set accordingly.
                // Let's derive from scratch for clarity.
                // We'll use cublasSgemmEx? No, just standard.
                // Define row-major matrices: A (M x K), B (K x N), C = A * B (M x N).
                // In column-major format, the same memory represents A_col = A^T (K x M) with leading dimension M? Actually if A is row-major MxK, its data is laid out in rows. For column-major, a matrix of dimensions p x q has element (i,j) at memory i + j*p. If we want to interpret A's memory as column-major with dimensions (p,q), then we need p*q = M*K and mapping. Usually we interpret A (MxK row-major) as column-major (K x M) with leading dimension M. Because element (i,j) row-major at i*K + j. As column-major with dimensions K x M, element (j,i) is at j + i*K, which matches i*K+j if we treat the column-major matrix as having K rows and M columns, and leading dimension K? Wait, column-major: element (r,c) at r + c*lda. If we set lda = K, then element (r,c) at r + c*K. But we want i*K+j, where i is row of A row-major, j is col. So if we set r = j (col index becomes row), c = i (row index becomes col), then element (r,c) = j + i*K, which is i*K+j. Perfect. So A row-major (MxK) can be viewed as column-major with dimensions (K x M) and leading dimension K, where the first dimension K corresponds to the original columns, and second dimension M corresponds to original rows. Similarly, B row-major (KxN) can be viewed as column-major with dimensions (N x K) and leading dimension N.
                // Then C row-major (MxN) viewed as column-major with dimensions (N x M) and leading dimension N.
                // Now we want C_col = B_col * A_col (since C = A*B, so C^T = B^T * A^T).
                // So call cublasSgemm with column-major, no transposes:
                // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                //            m = N, n = M, k = K,
                //            &alpha, B_col_pointer (actually B pointer), ldb = N,
                //            A_col_pointer (A pointer), lda = K,
                //            &beta, C_col_pointer (C pointer), ldc = N)
                // Here m = number of rows of C_col = N, n = number of cols of C_col = M, k = common dimension = K.
                // This is the standard technique for row-major GEMM without transposes.
                // So we can use this approach.

                // Step 1: in_projection: C = x * in_proj_weight^T.
                // Let A = x (M x K) with M = B*L, K = E.
                // Let B = in_proj_weight (N x K) with N = 3E, K = E. We need C = x * B^T = A * B^T, which is A times transpose of B. So the original B is N x K, and we want its transpose.
                // Using the column-major interpretation: A_col = A^T (K x M), B_col = B^T (K x N).
                // We want C = A * B^T, so C^T = (B^T)^T * A^T = B * A^T.
                // In column-major, C_col = C^T (N x M), and we can compute C_col = B_col^T * A_col^T? This is getting messy.
                // Alternative: use the transpose flag in cuBLAS to handle the transpose easily.
                // Let's use the standard approach with CUBLAS_OP_T for one matrix.
                // Usually for row-major, to compute C = A * B, we do:
                // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, B, N, A, K, &beta, C, N)
                // where A is MxK, B is KxN, and the first argument after handle are N, M, K.
                // Let's test with a simple example in mind: A 2x3, B 3x4, C 2x4.
                // We'll code it as: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 4, 2, 3, &alpha, B, 4, A, 3, &beta, C, 4).
                // This is known to work for row-major matrices stored contiguously.
                // So for in_projection, we want C = x * in_proj_weight^T. That means the second matrix is the transpose of in_proj_weight.
                // If we treat in_proj_weight as the "B" in the trick, it should be the matrix that we multiply, and we want its transpose.
                // So we can set: A = x (M x K), but in the trick the first matrix is B and second is A, with transposes. So maybe we need to adjust.
                // Actually the trick with both CUBLAS_OP_T computes: C = B^T * A^T? No.
                // The general rule: To compute C = A * B for row-major A (MxK), B (KxN), use:
                // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N)
                // where the first two dimensions are N and M, and the pointers are swapped. That is, pass B as the first matrix, A as second, and dimensions N, M, K.
                // This is equivalent to column-major multiplication of B^T and A^T.
                // So for C = A * B^T, we can think of it as A * (B^T), so B in this formula should be B^T. But we have B (NxK), so B^T is KxN. The formula expects B (KxN). So we have B^T already? We have W = in_proj_weight (3E x E). Its transpose is E x 3E. We want x (B*L x E) * W^T (E x 3E). So A = x (MxK with M=B*L, K=E), B = W^T (K x N with K=E, N=3E). But we have W not W^T. So we need to provide B as W^T. We can either transpose W beforehand (not good) or use the trick with transposes.
                // I think the simplest correct approach is to use cublasSgemm with CUBLAS_OP_T for the second matrix, while treating matrices as column-major with swapped dimensions.
                // Let's derive using column-major interpretation carefully.
                // Let memory of x be interpreted as column-major with dimensions (E, B*L) (since x row-major B*L x E -> column-major E x B*L). Then x_col = x^T (E x B*L).
                // in_proj_weight memory (row-major 3E x E) -> column-major E x 3E, so it's W_col = W^T (E x 3E).
                // We want C_row = x_row * W_row^T = x_row * (W_row)^T. In column-major, we have x_col = x_row^T, W_col = W_row^T. Then C_row^T = W_row * x_row^T = (W_col)^T * x_col.
                // So C_col = (C_row)^T = (W_col)^T * x_col.
                // Now we can use column-major GEMM with no transposes: C_col = (W_col)^T * x_col.
                // But (W_col)^T means we need to transpose W_col, which is the same as using CUBLAS_OP_T on W_col.
                // So we call cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                //            m = 3E, n = B*L, k = E,
                //            &alpha, W_col (in_proj_weight pointer), lda = E (since W_col is E x 3E, leading dim E? Wait column-major: for W_col dimensions (E x 3E), leading dimension is E, because column-major element (i,j) at i + j*lda, with lda = E. In memory, W_row is 3E x E, so W_col as E x 3E uses lda = E. Yes.
                //            x_col pointer (x), ldb = E (since x_col is E x B*L, leading dim E),
                //            &beta, C_col pointer (in_proj_output), ldc = 3E (since C_col is 3E x B*L, leading dim 3E).
                // This will compute C_col, which is the column-major representation of C_row (B*L x 3E) but stored as column-major with dimensions 3E x B*L, leading dim 3E. However, our output pointer is intended for row-major B*L x 3E. Does the memory layout match?
                // C_col has shape 3E x B*L, leading dim 3E, so its memory layout is: element (r,c) at r + c*3E. This is exactly the layout of C_row with shape B*L x 3E if we interpret C_row as column-major with shape 3E x B*L. That is, C_row (B*L x 3E) row-major memory is element (i,j) at i*3E + j. If we treat that same memory as column-major with shape 3E x B*L and leading dim 3E, element (j,i) is at j + i*3E = i*3E + j. So yes, the memory layout is identical. So C_col computed above is exactly the row-major C_row we want, stored in in_proj_output. Great.
                // So we can use this approach.
                // Thus, for in_projection:
                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            3 * E, B * L, E,
                            &alpha,
                            in_proj_weight, E,
                            x, E,
                            &beta,
                            in_proj_output, 3 * E);

                // Add bias: in_proj_output += in_proj_bias (broadcast over B*L)
                // Kernel to add bias row-wise.
                // We'll write a simple kernel.

                // Then split in_proj_output into q,k,v and permute to [N,L,D].
                // in_proj_output is [B*L, 3E] (row-major). We need to reshape to [B, L, 3E] and then split into q,k,v and further reshape/permute.
                // To avoid multiple kernels, we can do the split and permute in a single kernel, writing q,k,v in the desired layout [N, L, D] where N = B*H.
                // The desired layout for q: [N, L, D] with N = B*H, and within each head, the sequence dimension and head dimension are contiguous? For attention, we will use GEMM later, so we need q as [N, L, D] row-major (N rows of L*D).
                // Original in_proj_output layout: [B, L, 3E] with E = H*D. The three parts q,k,v are interleaved in the last dimension: for each position (b,l), the first E elements are q, next E are k, last E are v.
                // We need to map to q[n, l, d] where n = b*H + h, and within q part, the head index h selects a block of D from the E elements.
                // Specifically, q[b,l,h,d] = in_proj_output[b,l,h*D + d].
                // Desired memory layout for q: q[N, L, D] row-major, so element (n,l,d) at n*L*D + l*D + d.
                // So we can write a kernel that iterates over all output elements and computes source index.
                // Similarly for k and v with offsets E and 2E.

                // We'll launch a kernel to perform the permutation and also add bias.

                // For efficiency, we can combine bias addition and permutation in one kernel, but separate is fine.

                // Step 2: Compute Q*K^T scaled by 1/sqrt(D). We'll use GEMM for each head? Since Q is [N, L, D], K is [N, L, D], we want scores[n, i, j] = sum_d Q[n,i,d] * K[n,j,d] / sqrt(D).
                // This is essentially batched GEMM. We can use cublasSgemmStridedBatched or just loop over n and call cublasSgemm for each n. Since N=1024 (B=256, H=4), L=32, D=128. Batched GEMM would be efficient.
                // We'll use cublasSgemmStridedBatched with appropriate strides.
                // For each n, we compute a (L x D) * (D x L) matrix multiplication, giving L x L.
                // We'll use column-major trick again. We want scores_n (LxL) = Q_n (LxD) * K_n^T (DxL). So it's similar to in_projection but with smaller dimensions.
                // Using the column-major trick: compute scores_col = (K_n^T)^T * Q_n^T = K_n * Q_n^T.
                // We can set up cublasSgemmStridedBatched with appropriate transpose flags and strides.
                // But maybe simpler to launch a custom kernel for QK^T since D=128, which is small and we could use shared memory. But cuBLAS GEMM is likely faster.
                // We'll use cublasSgemmStridedBatched.

                // First, allocate scores [N, L, L].
                // For each n, we want to compute scores_n = Q_n * K_n^T * scale.
                // Using cublasSgemmStridedBatched with column-major as above:
                // We'll treat Q and K as column-major with shapes D x L (for Q) and D x L (for K) but we need K^T which is L x D. Actually we want scores (LxL) = Q (LxD) * K^T (DxL).
                // In column-major, Q_col = Q^T (D x L), K_col = K^T (D x L). Then scores_col = scores^T (L x L) = K_col^T * Q_col.
                // So again: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                //      m = L, n = L, k = D,
                //      &alpha, K_col, lda = D,
                //      Q_col, ldb = D,
                //      &beta, scores_col, ldc = L)
                // But we need scale by 1/sqrt(D). We can set alpha = INV_SQRT_DIM.
                // For strided batched, we need to set the strides for each matrix correctly.

                // Since N is large, we can use cublasSgemmStridedBatched.
                // Define strides:
                // Q: base pointer q, strideQ = L*D (distance between n)
                // K: base pointer k, strideK = L*D
                // scores: base pointer scores, strideScores = L*L
                // Then call cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                //      L, L, D, &alpha, k, D, L*D, q, D, L*D, &beta, scores, L, L*L, N)

                // After that, apply softmax on scores for each n and row i.

                // Step 3: Compute attention output: attn_out = softmax_scores * V.
                // attn_out_n (LxD) = softmax_scores_n (LxL) * V_n (LxD).
                // Again batched GEMM. This time no transpose on V, but we have softmax_scores on left.
                // Use column-major trick: attn_out_col = V_col^T * softmax_scores_col^T? Let's derive.
                // Want attn_out (LxD) = P (LxL) * V (LxD).
                // Column-major: P_col = P^T (LxL), V_col = V^T (DxL), attn_out_col = attn_out^T (D x L).
                // attn_out_col = V_col^T * P_col.
                // So we call cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                //      D, L, L, &alpha, v, D, L*D, softmax_scores, L, L*L, &beta, attn_out, D, L*D, N)

                // Step 4: Concatenate heads and apply out_projection.
                // attn_out is [N, L, D] row-major. We need to reshape to [B, L, E] with head dimension concatenated: out[b,l,h*D+d] = attn_out[b*H+h, l, d].
                // We can write a kernel to do this permutation into attn_out_reshaped [B, L, E].
                // Then out_projection: final = attn_out_reshaped @ out_proj_weight^T + out_proj_bias.
                // attn_out_reshaped is [B*L, E] row-major, out_proj_weight is [E, E]. So final = attn_out_reshaped * out_proj_weight^T.
                // Use same GEMM approach as in_projection but with different dimensions.
                // We'll allocate final output already provided as attn_output pointer.

                // Additionally, compute avg attention weights: attn_weights_avg [L, B, B]? Actually shape [seq_len, batch, batch] = [32,256,256].
                // We need to average softmax_scores over heads: for each b, i, j, attn_weights_avg[i,b,j]? Contract says shape [32,256,256] which is [seq_len, batch, batch]. Since batch=256, seq_len=32. So attn_weights_avg shape is [L, B, B] (first dimension L=32, then B=256, B=256). But softmax_scores shape is [N, L, L] with N=B*H. The averaging over heads means for each batch b and each i,j, average over h: avg[b,i,j] = (1/H) * sum_h softmax_scores[b*H+h, i, j]. Then we need to transpose the batch and seq dimensions: attn_weights_avg[i,b,j] = avg[b,i,j]. So output shape [L, B, B].
                // We'll write a kernel to compute this average and transpose.

    // Let's implement the kernels for bias add, permute, and averaging.
    // We'll also need a kernel to add bias to in_proj_output.

    // Bias add kernel: in_proj_output[row, col] += in_proj_bias[col] for each row in B*L, col in 3E.
    __global__ void add_bias_kernel(float* data, const float* bias, int rows, int cols) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols) {
            int row = idx / cols;
            int col = idx % cols;
            data[idx] += bias[col];
        }
    }

    // Permute kernel: from in_proj_output [B, L, 3E] to q,k,v [N, L, D] and also combine with bias? Bias already added, so just permute.
    __global__ void permute_qkv_kernel(const float* in_proj, float* q, float* k, float* v,
                                       int B, int L, int E, int H, int D) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * L * 3 * E;
        if (idx < total) {
            int b = idx / (L * 3 * E);
            int rem = idx % (L * 3 * E);
            int l = rem / (3 * E);
            int rem2 = rem % (3 * E);
            int part = rem2 / E; // 0 for q, 1 for k, 2 for v
            int h = (rem2 % E) / D;
            int d = (rem2 % E) % D;
            int n = b * H + h;
            int out_idx = n * L * D + l * D + d;
            if (part == 0) q[out_idx] = in_proj[idx];
            else if (part == 1) k[out_idx] = in_proj[idx];
            else v[out_idx] = in_proj[idx];
        }
    }

    // Concatenate heads back: from attn_out [N, L, D] to attn_out_reshaped [B, L, E]
    __global__ void concat_heads_kernel(const float* attn_out, float* reshaped,
                                        int B, int L, int E, int H, int D) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * L * E;
        if (idx < total) {
            int b = idx / (L * E);
            int rem = idx % (L * E);
            int l = rem / E;
            int e = rem % E;
            int h = e / D;
            int d = e % D;
            int n = b * H + h;
            reshaped[idx] = attn_out[n * L * D + l * D + d];
        }
    }

    // Average attention weights over heads and transpose to [L, B, B]
    __global__ void average_attn_weights_kernel(const float* softmax_scores, float* attn_weights_avg,
                                                int B, int L, int H) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = L * B * B;
        if (idx < total) {
            int i = idx / (B * B);
            int rem = idx % (B * B);
            int b_row = rem / B;
            int b_col = rem % B;
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                int n = b_row * H + h;
                sum += softmax_scores[n * L * L + i * L + b_col];
            }
            attn_weights_avg[idx] = sum / H;
        }
    }

    // Now inside launcher, after allocating memory, launch the sequence of operations.
    // We'll use a helper to launch kernels with grid size.
    auto launch_kernel = [&](dim3 grid, dim3 block, void* kernel, void** args) {
        // This is not straightforward; we'll just launch directly.
    };

    // We'll write direct launches.

    // 1. GEMM for in_projection
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                3 * E, B * L, E,
                &alpha,
                in_proj_weight, E,
                x, E,
                &beta,
                in_proj_output, 3 * E);

    // 2. Add bias
    {
        int block = 256;
        int grid = (B * L * 3 * E + block - 1) / block;
        add_bias_kernel<<<grid, block, 0, stream>>>(in_proj_output, in_proj_bias, B * L, 3 * E);
    }

    // 3. Permute to q,k,v
    {
        int block = 256;
        int grid = (B * L * 3 * E + block - 1) / block;
        permute_qkv_kernel<<<grid, block, 0, stream>>>(in_proj_output, q, k, v, B, L, E, H, D);
    }

    // 4. QK^T scaled (batched GEMM)
    float scale = INV_SQRT_DIM;
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              L, L, D,
                              &scale,
                              k, D, L * D,
                              q, D, L * D,
                              &beta, // beta=0
                              scores, L, L * L,
                              N);

    // 5. Softmax over scores (in-place or with separate buffer)
    {
        int block = 256;
        int grid = (N * L + block - 1) / block; // each thread handles one row? Actually softmax_kernel handles one row per thread, but we set total = N*L (number of rows).
        softmax_kernel<<<grid, block, 0, stream>>>(softmax_scores, scores, N, L);
    }

    // 6. attention output: attn_out = softmax_scores * V
    float alpha_attn = 1.0f;
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              D, L, L,
                              &alpha_attn,
                              v, D, L * D,
                              softmax_scores, L, L * L,
                              &beta,
                              attn_out, D, L * D,
                              N);

    // 7. Concatenate heads
    {
        int block = 256;
        int grid = (B * L * E + block - 1) / block;
        concat_heads_kernel<<<grid, block, 0, stream>>>(attn_out, attn_out_reshaped, B, L, E, H, D);
    }

    // 8. out_projection: final = attn_out_reshaped @ out_proj_weight^T + out_proj_bias
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, B * L, E,
                &alpha,
                out_proj_weight, E,
                attn_out_reshaped, E,
                &beta,
                attn_output, E);

    // Add out_proj_bias
    {
        int block = 256;
        int grid = (B * L * E + block - 1) / block;
        add_bias_kernel<<<grid, block, 0, stream>>>(attn_output, out_proj_bias, B * L, E);
    }

    // 9. Average attention weights
    {
        int block = 256;
        int grid = (L * B * B + block - 1) / block;
        average_attn_weights_kernel<<<grid, block, 0, stream>>>(softmax_scores, attn_weights_avg, B, L, H);
    }

    // Free temporary memory
    cudaFree(in_proj_output);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(scores);
    cudaFree(softmax_scores);
    cudaFree(attn_out);
    cudaFree(attn_out_reshaped);

    cublasDestroy(handle);
}
