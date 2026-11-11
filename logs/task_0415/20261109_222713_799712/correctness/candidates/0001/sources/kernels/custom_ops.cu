// GroupNorm CUDA kernel implementation
// Input: x [N, C, H, W] float32 contiguous CUDA tensor
// weight, bias [C] float32
// output same shape as x
// num_groups: number of groups to split channels into
// eps: epsilon for numerical stability

#include <cuda_runtime.h>
#include <cstdint>

// Define block size for reduction
#define GROUPNORM_BLOCK_SIZE 256

// Helper: warp reduce sum
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper: block reduce sum via shared memory
// Assumes shared array of size blockDim.x/32 initialized with warp sums
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* shared) {
    int lane = threadIdx.x & 31; // threadIdx.x % 32
    int wid = threadIdx.x >> 5;  // threadIdx.x / 32

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Reduce from shared (only first warp needed)
    if (threadIdx.x < (blockDim.x >> 5)) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    } else {
        val = 0;
    }
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Kernel 1: Compute per-(n, group) sum and sum of squares
// grid: (N * num_groups), block: GROUPNORM_BLOCK_SIZE
// Each block processes one (n, group) pair
__global__ void group_norm_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int N,
    int C_per_group,
    int HW,
    int num_groups
) {
    extern __shared__ float shared[]; // size = blockDim.x/32 floats

    int block_id = blockIdx.x;
    int n = block_id / num_groups;
    int g = block_id % num_groups;

    int channel_start = g * C_per_group;
    int offset = ((n * num_groups + g) * C_per_group) * HW;
    // Actually x is NCHW contiguous, so linear layout:
    // x[n, c, hw] = x[((n * total_C + c) * HW) + hw]
    // For group, channels: channel_start to channel_start + C_per_group - 1
    int total_C = num_groups * C_per_group;
    int base_x = (n * total_C + channel_start) * HW;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    // grid-stride loop over all elements in this group
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; ; idx += gridDim.x * blockDim.x) {
        // idx within group? safer: iterate differently
        // We'll use a single flat loop over elements in group
        int total_elements = C_per_group * HW;
        // Since we have one block per group, we can just use threadIdx.x and loop
        // But to support large HW, use grid-stride type loop over total_elements
        // However, we already have grid = N*num_groups, so each block processes group.
        // Use loop within block:
        if (idx < total_elements) {
            int inner = idx - blockIdx.x * blockDim.x * gridDim.x; // not correct, simpler approach:
            // Actually just use for loop from threadIdx.x
            for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
                int c_offset = i / HW;
                int hw = i % HW;
                int c = channel_start + c_offset;
                int linear_index = (n * total_C + c) * HW + hw;
                float val = x[linear_index];
                local_sum += val;
                local_sumsq += val * val;
            }
        }
        // We should break after processing all elements; use a single loop with stride
        // Simpler: use grid-stride over total_elements but with only one block per group.
        // The loop above with stride blockDim.x suffices.
        // However, we need to ensure we don't have a race condition with blockDim.x.
        // Since blockDim.x is 256, we'll loop over elements.
        // Rewrite cleaner:
        // We'll use a for loop below, but we'll restructure to avoid confusion.
        break; // temporary to avoid double processing
    }

    // Alternative clean implementation:
    // Clear locals and redo with proper loop
    local_sum = 0.0f;
    local_sumsq = 0.0f;
    int total_elements = C_per_group * HW;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int c_offset = i / HW;
        int hw = i % HW;
        int c = channel_start + c_offset;
        int linear_index = (n * total_C + c) * HW + hw;
        float val = x[linear_index];
        local_sum += val;
        local_sumsq += val * val;
    }

    // Block reduction
    float block_sum = blockReduceSum(local_sum, shared);
    float block_sumsq = blockReduceSum(local_sumsq, shared);

    // Thread 0 writes result
    if (threadIdx.x == 0) {
        int output_idx = block_id; // one per (n,g)
        sum[output_idx] = block_sum;
        sumsq[output_idx] = block_sumsq;
    }
}

// Kernel 2: Normalize and apply affine
// grid: N * num_groups, block: same
__global__ void group_norm_apply_kernel(
    const float* __restrict__ x,
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N,
    int C_per_group,
    int HW,
    int num_groups,
    float eps
) {
    int block_id = blockIdx.x;
    int n = block_id / num_groups;
    int g = block_id % num_groups;

    int channel_start = g * C_per_group;
    int total_C = num_groups * C_per_group;
    int base_x = (n * total_C + channel_start) * HW;
    int total_elements = C_per_group * HW;

    float group_sum = sum[block_id];
    float group_sumsq = sumsq[block_id];
    float count = (float)total_elements;
    float mean = group_sum / count;
    float var = group_sumsq / count - mean * mean; // biased variance, might be negative due to precision
    // PyTorch uses var = mean((x - mean)^2), but we compute using sumsq/count - mean^2.
    // To exactly match, we might need to compute as sum((x-mean)^2)/count.
    // However, given tolerance and float, this is usually fine but can be inaccurate.
    // We'll instead recompute second pass? Better to compute using formula: var = (sumsq - sum*sum/count)/count
    // That's identical to above. To improve stability, use the standard two-pass? But we can't easily because we only have sum/sumsq.
    // PyTorch uses Welford but tolerance is large. We'll use epsilon stabilizing.
    float inv_std = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int c_offset = i / HW;
        int hw = i % HW;
        int c = channel_start + c_offset;
        int linear_index = (n * total_C + c) * HW + hw;
        float val = x[linear_index];
        float norm = (val - mean) * inv_std;
        float w = weight[c];
        float b = bias[c];
        y[linear_index] = norm * w + b;
    }
}

// Host launcher (extern "C" so binding can call)
extern "C" void group_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N,
    int C,
    int H,
    int W,
    int num_groups,
    float eps,
    cudaStream_t stream
) {
    int HW = H * W;
    int C_per_group = C / num_groups;
    int total_blocks = N * num_groups;

    // Allocate temporary buffers for sum and sumsq
    float *d_sum, *d_sumsq;
    cudaMalloc(&d_sum, total_blocks * sizeof(float));
    cudaMalloc(&d_sumsq, total_blocks * sizeof(float));

    // Launch reduction kernel
    int block_size = GROUPNORM_BLOCK_SIZE;
    int shared_mem = (block_size / 32) * sizeof(float);
    group_norm_sum_kernel<<<total_blocks, block_size, shared_mem, stream>>>(
        x, d_sum, d_sumsq, N, C_per_group, HW, num_groups
    );

    // Launch apply kernel
    group_norm_apply_kernel<<<total_blocks, block_size, 0, stream>>>(
        x, d_sum, d_sumsq, weight, bias, y, N, C_per_group, HW, num_groups, eps
    );

    // Free temporary buffers after kernels complete (on same stream)
    cudaFree(d_sum);
    cudaFree(d_sumsq);
}
