#include <cuda_runtime.h>
#include <math.h>

// BatchNorm2d evaluation mode kernel
// y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Broadcast over N, H, W dimensions.
__global__ void batch_norm_eval_2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N,
    int C,
    int HW)
{
    int64_t total = (int64_t)N * C * HW;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int c = (idx / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;
        float val = (x[idx] - mean) / sqrtf(var + eps) * w + b;
        y[idx] = val;
    }
}

// Triplet margin loss with Lp distance (p=2), reduction='sum'
// For each sample n: d_pos = sqrt(sum((anchor - positive)^2)), d_neg = sqrt(sum((anchor - negative)^2))
// loss_n = max(d_pos - d_neg + margin, 0)
// total_loss = sum(loss_n)
// This kernel uses one block per sample and a robust block reduction.
__global__ void triplet_margin_loss_sum_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float margin,
    int N,
    int D,
    float* __restrict__ output)  // output is a scalar tensor, we use atomicAdd
{
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Each thread accumulates partial sums over a strided subset of D elements
    float pos_sum = 0.0f;
    float neg_sum = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        int64_t idx = (int64_t)n * D + i;
        float diff_pos = anchor[idx] - positive[idx];
        float diff_neg = anchor[idx] - negative[idx];
        pos_sum += diff_pos * diff_pos;
        neg_sum += diff_neg * diff_neg;
    }

    // Block reduction for pos_sum and neg_sum using warp shuffles and shared memory
    // First, warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        pos_sum += __shfl_down_sync(0xffffffff, pos_sum, offset);
        neg_sum += __shfl_down_sync(0xffffffff, neg_sum, offset);
    }

    // Shared memory for per-warp results
    __shared__ float warp_pos[32]; // max 32 warps (block size <= 1024)
    __shared__ float warp_neg[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (block_size + 31) / 32;

    if (lane_id == 0) {
        warp_pos[warp_id] = pos_sum;
        warp_neg[warp_id] = neg_sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp sums
    if (warp_id == 0) {
        // Only threads with lane_id < num_warps participate
        if (lane_id < num_warps) {
            float p = warp_pos[lane_id];
            float q = warp_neg[lane_id];
            // Warp reduction within first warp
            for (int offset = 16; offset > 0; offset >>= 1) {
                p += __shfl_down_sync(0xffffffff, p, offset);
                q += __shfl_down_sync(0xffffffff, q, offset);
            }
            if (lane_id == 0) {
                float d_pos = sqrtf(p);
                float d_neg = sqrtf(q);
                float loss = fmaxf(d_pos - d_neg + margin, 0.0f);
                atomicAdd(output, loss);
            }
        }
    }
}

// Launcher for triplet_margin_loss_sum
// Returns sum of losses across batch as a single float scalar, stored in output[0]
extern "C" void triplet_margin_loss_sum_atomic_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float margin,
    int p,
    int N,
    int D,
    float* output,
    cudaStream_t stream)
{
    // Zero output before atomic accumulation
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = N; // one block per sample

    triplet_margin_loss_sum_kernel<<<blocks, threads, 0, stream>>>(
        anchor, positive, negative, margin, N, D, output);
}

extern "C" void batch_norm_eval_2d_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream)
{
    int HW = H * W;
    int64_t total = (int64_t)N * C * HW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    batch_norm_eval_2d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var, eps, N, C, HW);
}
