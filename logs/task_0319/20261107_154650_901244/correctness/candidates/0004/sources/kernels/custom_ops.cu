// CUDA implementation of Multihead Attention forward
// This file contains CUDA kernels and launchers for the mha_forward operator.
// Dependencies: CUDA runtime and cuBLAS (for GEMM).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Define constants for clarity
constexpr int WARP_SIZE = 32;
// INV_SQRT_DIM cannot be constexpr because sqrtf is not constexpr in CUDA C++.
// Use a global constant instead; it is evaluated at runtime, which is fine here.
const float INV_SQRT_DIM = 1.0f / sqrtf(128.0f); // 1/sqrt(head_dim)

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

// Add bias kernel: data[row, col] += bias[col]
__global__ void add_bias_kernel(float* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        data[idx] += bias[col];
    }
}

// Permute kernel: from in_proj_output [B, L, 3E] to q,k,v [N, L, D]
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

// Launcher function for mha_forward.
// Input tensors are assumed to be on GPU, contiguous for weights, x may be non-contiguous but we can assume contiguous for simplicity
// because binding layer can make it contiguous if needed.
// Shapes: x: [batch, seq_len, embed_dim] = [256,32,512]
// in_proj_weight: [3*embed_dim, embed_dim] = [1536,512]
// in_proj_bias: [1536]
// out_proj_weight: [embed_dim, embed_dim] = [512,512]
// out_proj_bias: [512]
// Outputs: attn_output [batch, seq_len, embed_dim] = [256,32,512]
//          attn_weights_avg [seq_len, batch, batch] = [32,256,256]

extern "C" void mha_forward_launcher(
    const float* x,             // [B, L, E]
    const float* in_proj_weight,// [3E, E]
    const float* in_proj_bias,  // [3E]
    const float* out_proj_weight,// [E, E]
    const float* out_proj_bias, // [E]
    float* attn_output,         // [B, L, E]
    float* attn_weights_avg,    // [L, B, B]
    int B, int L, int E, int H, // H = num_heads = 4
    cudaStream_t stream)
{
    // Dimensions
    int N = B * H;          // total number of heads across batch
    int D = E / H;          // head dimension = 512/4 = 128

    // Workspace sizes
    float *in_proj_output, *q, *k, *v, *scores, *softmax_scores, *attn_out, *attn_out_reshaped;
    auto checkCuda = [&](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
            // no cleanup for now, but we can set global error flag? just throw?
            // Not throwing here to avoid complexity, but log.
        }
    };
    checkCuda(cudaMalloc(&in_proj_output, (size_t)B * L * 3 * E * sizeof(float)), "cudaMalloc in_proj_output");
    checkCuda(cudaMalloc(&q, (size_t)N * L * D * sizeof(float)), "cudaMalloc q");
    checkCuda(cudaMalloc(&k, (size_t)N * L * D * sizeof(float)), "cudaMalloc k");
    checkCuda(cudaMalloc(&v, (size_t)N * L * D * sizeof(float)), "cudaMalloc v");
    checkCuda(cudaMalloc(&scores, (size_t)N * L * L * sizeof(float)), "cudaMalloc scores");
    checkCuda(cudaMalloc(&softmax_scores, (size_t)N * L * L * sizeof(float)), "cudaMalloc softmax_scores");
    checkCuda(cudaMalloc(&attn_out, (size_t)N * L * D * sizeof(float)), "cudaMalloc attn_out");
    checkCuda(cudaMalloc(&attn_out_reshaped, (size_t)B * L * E * sizeof(float)), "cudaMalloc attn_out_reshaped");

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t cub_status = cublasCreate(&handle);
    if (cub_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS init failed\n");
        return;
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Step 1: in_projection: proj = x @ in_proj_weight^T + in_proj_bias
    // x: [B*L, E] row-major; in_proj_weight: [3E, E] row-major
    // We need C = x (B*L x E) * W^T (E x 3E)
    // Using column-major interpretation: W_col = W^T (E x 3E), x_col = x^T (E x B*L)
    // C_col = C^T (3E x B*L) = W_col^T * x_col
    // So call cublasSgemm with CUBLAS_OP_T on W, CUBLAS_OP_N on x, dimensions m=3E, n=B*L, k=E.
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                3 * E, B * L, E,
                &alpha,
                in_proj_weight, E,
                x, E,
                &beta,
                in_proj_output, 3 * E);

    // Step 2: Add bias to in_proj_output
    {
        int block = 256;
        int grid = (B * L * 3 * E + block - 1) / block;
        add_bias_kernel<<<grid, block, 0, stream>>>(in_proj_output, in_proj_bias, B * L, 3 * E);
    }

    // Step 3: Permute to q, k, v
    {
        int block = 256;
        int grid = (B * L * 3 * E + block - 1) / block;
        permute_qkv_kernel<<<grid, block, 0, stream>>>(in_proj_output, q, k, v, B, L, E, H, D);
    }

    // Step 4: QK^T scaled (batched GEMM)
    float scale = INV_SQRT_DIM;
    // FIX: leading dimension for transposed k should be L, not D.
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              L, L, D,
                              &scale,
                              k, L, L * D,   // A transposed, leading dim L, stride L*D
                              q, D, L * D,   // B not transposed, leading dim D, stride L*D
                              &beta,
                              scores, L, L * L,
                              N);

    // Step 5: Softmax over scores (each row per thread)
    {
        int block = 256;
        int grid = (N * L + block - 1) / block;
        softmax_kernel<<<grid, block, 0, stream>>>(softmax_scores, scores, N, L);
    }

    // Step 6: attention output: attn_out = softmax_scores * V
    float alpha_attn = 1.0f;
    // FIX: leading dimension for transposed v should be L, not D.
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              D, L, L,
                              &alpha_attn,
                              v, L, L * D,   // A transposed, leading dim L, stride L*D
                              softmax_scores, L, L * L, // B not transposed, leading dim L, stride L*L
                              &beta,
                              attn_out, D, L * D,
                              N);

    // Step 7: Concatenate heads
    {
        int block = 256;
        int grid = (B * L * E + block - 1) / block;
        concat_heads_kernel<<<grid, block, 0, stream>>>(attn_out, attn_out_reshaped, B, L, E, H, D);
    }

    // Step 8: out_projection: final = attn_out_reshaped @ out_proj_weight^T + out_proj_bias
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, B * L, E,
                &alpha,
                out_proj_weight, E,
                attn_out_reshaped, E,
                &beta,
                attn_output, E);

    // Step 9: Add out_proj_bias
    {
        int block = 256;
        int grid = (B * L * E + block - 1) / block;
        add_bias_kernel<<<grid, block, 0, stream>>>(attn_output, out_proj_bias, B * L, E);
    }

    // Step 10: Average attention weights
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