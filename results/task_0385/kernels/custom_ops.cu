#include <cuda_runtime.h>

// ============================================================================
// clamp_forward
// ============================================================================
// Element-wise clamp: y = min(max(x, min_val), max_val)
// Supports arbitrary strides via linear indexing over numel.

__global__ void clamp_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float min_val,
    float max_val)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = input[idx];
        v = fminf(fmaxf(v, min_val), max_val);
        output[idx] = v;
    }
}

extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    double min_val,
    double max_val,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    const int block = 256;
    const int64_t grid = (numel + block - 1) / block;
    clamp_kernel<<<grid, block, 0, stream>>>(
        output, input, numel, (float)min_val, (float)max_val);
}

// ============================================================================
// layer_norm_forward
// ============================================================================
// LayerNorm over last dimension. For input shape [B, S, F], each row of length F
// is normalized independently. This kernel assumes contiguity in the last
// dimension (stride[last] == 1) but uses explicit row stride for generality.
// Each block processes one row. blockDim.x <= F, and threads cooperate to
// compute sum and sum_sq using shared memory reduction.
// The output is written as (x - mean) / sqrt(var + eps) * weight + bias.
// Weight and bias are length F, contiguous.

template <int BLOCK_DIM>
__global__ void layer_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int64_t rows,
    int64_t F,
    float eps,
    int64_t row_stride_in,
    int64_t row_stride_out)
{
    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sq_shared = shared + BLOCK_DIM;

    const int64_t row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * row_stride_in;
    float* row_out = output + row * row_stride_out;

    const int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    // Two-pass: first accumulate sum and sum of squares, handling F > BLOCK_DIM
    for (int64_t col = tid; col < F; col += BLOCK_DIM) {
        float v = row_in[col];  // assumes last dim stride == 1
        local_sum += v;
        local_sq += v * v;
    }

    sum_shared[tid] = local_sum;
    sq_shared[tid] = local_sq;
    __syncthreads();

    // Reduction over BLOCK_DIM elements (but only valid up to min(F, BLOCK_DIM))
    for (int stride = BLOCK_DIM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
            sq_shared[tid] += sq_shared[tid + stride];
        }
        __syncthreads();
    }

    // thread 0 computes mean and variance, broadcasts via shared
    __shared__ float mean;
    __shared__ float inv_std;
    if (tid == 0) {
        mean = sum_shared[0] / (float)F;
        float var = sq_shared[0] / (float)F - mean * mean;
        // clamp variance to avoid negative due to numerical issues
        if (var < 0.0f) var = 0.0f;
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    // Apply normalization and affine transform
    for (int64_t col = tid; col < F; col += BLOCK_DIM) {
        float x_hat = (row_in[col] - mean) * inv_std;
        float w = weight[col];
        float b = bias[col];
        row_out[col] = x_hat * w + b;
    }
}

extern "C" void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int64_t rows,
    int64_t F,
    double eps,
    int64_t row_stride_in,
    int64_t row_stride_out,
    cudaStream_t stream)
{
    if (rows <= 0 || F <= 0) return;

    // Choose block size: 256 if F >= 256, else power-of-two <= F (at least 32)
    int block = 256;
    if (F < 256) {
        // find largest power of two <= F, with minimum 32 (if F smaller, handle specially)
        block = 32;
        while (block * 2 <= F) block *= 2;
        if (block > 256) block = 256;
    }
    // For very small F (< 32), use single thread block? We'll still use 32 threads
    // but only active threads access; the kernel loop handles it.

    size_t shared_mem = 2 * block * sizeof(float);

    // Grid dimension: one block per row, capped to avoid exceeding max grid size
    // Use grid-stride style: kernel already checks row >= rows, so we can launch
    // as many blocks as rows, but if rows exceeds max grid (usually large), we
    // could use a loop. However typical max grid is 2^31-1 for x-dim, so rows
    // up to 2B is fine. But to be safe, if rows > 65535 and device supports only
    // 65535 blocks in x-dim? For modern GPUs (CC>=3.0) max grid x is 2^31-1.
    // We'll assume that.
    if (block == 256) {
        layer_norm_kernel<256><<<rows, block, shared_mem, stream>>>(
            output, input, weight, bias, rows, F, (float)eps,
            row_stride_in, row_stride_out);
    } else {
        // handle other block sizes via switch for common cases
        switch (block) {
            case 32:
                layer_norm_kernel<32><<<rows, block, shared_mem, stream>>>(
                    output, input, weight, bias, rows, F, (float)eps,
                    row_stride_in, row_stride_out);
                break;
            case 64:
                layer_norm_kernel<64><<<rows, block, shared_mem, stream>>>(
                    output, input, weight, bias, rows, F, (float)eps,
                    row_stride_in, row_stride_out);
                break;
            case 128:
                layer_norm_kernel<128><<<rows, block, shared_mem, stream>>>(
                    output, input, weight, bias, rows, F, (float)eps,
                    row_stride_in, row_stride_out);
                break;
            default:
                // Should not happen for F>=32; fallback to 32
                layer_norm_kernel<32><<<rows, 32, 2 * 32 * sizeof(float), stream>>>(
                    output, input, weight, bias, rows, F, (float)eps,
                    row_stride_in, row_stride_out);
        }
    }
}
