#include <cuda_runtime.h>

// CUDA kernels for custom embedding and layer norm
// All launchers use raw pointers and stream; no torch/ATen/pybind includes.

// ---------------------------------------------------------------------------
// Kernel: embedding lookup
// Each thread handles multiple output elements via grid-stride loop.
// Input indices (int64) and weight (float32) are assumed to be in device memory.
// Output is float32 in device memory.
// ---------------------------------------------------------------------------
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_elements,   // batch * seq_len
    int64_t embedding_dim)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t pos = idx; pos < total_elements; pos += stride) {
        int64_t token = indices[pos];
        // Safe if indices are valid; no bounds check for speed, but could add if needed.
        const float* src_row = weight + token * embedding_dim;
        float* dst_row = output + pos * embedding_dim;
        for (int64_t d = 0; d < embedding_dim; ++d) {
            dst_row[d] = src_row[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: layer normalization over last dimension
// Each block handles one or more rows; within a block, threads cooperate to
// compute mean and variance for a row, then apply normalization and optionally
// affine transformation. We use a 1D thread block with size >= dim (must be power of 2).
// For dim=32, blockDim.x=32 is natural; we handle general dim up to 1024 with blockDim.x = next_pow2(dim).
// ---------------------------------------------------------------------------
template <int BLOCK_DIM>
__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,   // may be nullptr
    const float* __restrict__ bias,     // may be nullptr
    float* __restrict__ output,
    int64_t rows,
    int64_t dim,
    float eps,
    bool elementwise_affine)
{
    // Each block handles one row.
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * dim;
    float* row_out = output + row * dim;

    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float s_weight[BLOCK_DIM];
    __shared__ float s_bias[BLOCK_DIM];

    // Load affine parameters into shared memory if needed.
    if (elementwise_affine) {
        if (weight != nullptr) {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_weight[i] = weight[i];
            }
        } else {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_weight[i] = 1.0f;
            }
        }
        if (bias != nullptr) {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_bias[i] = bias[i];
            }
        } else {
            for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
                s_bias[i] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Compute sum and sum of squares using warp reductions.
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
        float v = row_in[i];
        local_sum += v;
        local_sq_sum += v * v;
    }

    // Block reduction
    __shared__ float s_sum[BLOCK_DIM / 32];
    __shared__ float s_sq_sum[BLOCK_DIM / 32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq_sum[warp_id] = local_sq_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        float total_sq_sum = 0.0f;
        int num_warps = BLOCK_DIM / 32;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += s_sum[i];
            total_sq_sum += s_sq_sum[i];
        }
        float mean = total_sum / dim;
        float var = total_sq_sum / dim - mean * mean;
        s_mean = mean;
        s_var = var;
    }
    __syncthreads();

    float mean = s_mean;
    float var = s_var;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine
    for (int i = threadIdx.x; i < dim; i += BLOCK_DIM) {
        float normalized = (row_in[i] - mean) * inv_std;
        if (elementwise_affine) {
            float w = s_weight[i];
            float b = s_bias[i];
            row_out[i] = normalized * w + b;
        } else {
            row_out[i] = normalized;
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher: custom_embedding
// indices: CPU int64 tensor pointer
// weight: CPU float32 tensor pointer
// output: GPU float32 tensor pointer (preallocated by caller)
// batch, seq_len, vocab_size, embedding_dim: dimensions
// stream: CUDA stream to use
// ---------------------------------------------------------------------------
extern "C" void custom_embedding_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t vocab_size,
    int64_t embedding_dim,
    cudaStream_t stream)
{
    int64_t total_positions = batch * seq_len;
    // Allocate temporary device arrays for indices and weight.
    int64_t* d_indices = nullptr;
    float* d_weight = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_indices, total_positions * sizeof(int64_t));
    if (err != cudaSuccess) return; // caller will check error? We rely on cudaGetLastError later.
    err = cudaMalloc(&d_weight, vocab_size * embedding_dim * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_indices);
        return;
    }

    // Copy asynchronously on provided stream.
    cudaMemcpyAsync(d_indices, indices, total_positions * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_weight, weight, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Launch kernel.
    int block_size = 256;
    int grid_size = (int)((total_positions + block_size - 1) / block_size);
    // Grid-stride loop handles larger sizes; cap grid if needed (already reasonable).
    embedding_kernel<<<grid_size, block_size, 0, stream>>>(d_indices, d_weight, output, total_positions, embedding_dim);

    // Free temporaries after kernel completes; using cudaFree is synchronous with stream.
    cudaFree(d_indices);
    cudaFree(d_weight);
    // Note: cudaFree is safe after kernel uses memory because it defers until completion.
    // However, to be safe we could use cudaStreamSynchronize(stream) before free, but cudaFree after launch is okay.
}

// ---------------------------------------------------------------------------
// Launcher: custom_layer_norm
// input: GPU float32 tensor pointer
// weight: CPU float32 pointer (may be nullptr)
// bias: CPU float32 pointer (may be nullptr)
// output: GPU float32 tensor pointer (same shape as input)
// rows: number of rows (batch*seq_len)
// dim: normalized dimension (last dim)
// eps: epsilon
// elementwise_affine: bool
// stream: CUDA stream
// ---------------------------------------------------------------------------
extern "C" void custom_layer_norm_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t rows,
    int64_t dim,
    double eps,
    bool elementwise_affine,
    cudaStream_t stream)
{
    cudaError_t err;
    float* d_weight = nullptr;
    float* d_bias = nullptr;

    if (elementwise_affine) {
        if (weight != nullptr) {
            err = cudaMalloc(&d_weight, dim * sizeof(float));
            if (err != cudaSuccess) return;
            cudaMemcpyAsync(d_weight, weight, dim * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
        if (bias != nullptr) {
            err = cudaMalloc(&d_bias, dim * sizeof(float));
            if (err != cudaSuccess) {
                cudaFree(d_weight);
                return;
            }
            cudaMemcpyAsync(d_bias, bias, dim * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    }

    // Determine block size: next power of two >= dim, but at least 32.
    int block_dim = 32;
    while (block_dim < dim) block_dim <<= 1;
    // Cap at 1024 (max threads per block).
    if (block_dim > 1024) block_dim = 1024;

    // Launch templated kernel for common sizes; for generality we use a switch up to 1024.
    // For dim=32, block_dim=32. For dim>1024, we would need a different approach, but contract uses 32.
    dim3 grid(rows); // one block per row
    switch (block_dim) {
        case 32:
            layer_norm_kernel<32><<<grid, 32, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 64:
            layer_norm_kernel<64><<<grid, 64, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 128:
            layer_norm_kernel<128><<<grid, 128, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 256:
            layer_norm_kernel<256><<<grid, 256, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 512:
            layer_norm_kernel<512><<<grid, 512, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        case 1024:
            layer_norm_kernel<1024><<<grid, 1024, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
            break;
        default:
            // Should not happen for dim <= 1024; use 32 as fallback but may fail for larger dim.
            layer_norm_kernel<32><<<grid, 32, 0, stream>>>(input, d_weight, d_bias, output, rows, dim, (float)eps, elementwise_affine);
    }

    // Free temporary memory after kernel uses it (cudaFree defers).
    if (d_weight) cudaFree(d_weight);
    if (d_bias) cudaFree(d_bias);
}
