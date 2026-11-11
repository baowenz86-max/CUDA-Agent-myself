#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Kernel: mean over dimension 0 with keepdim=True
// Input: x with shape [B, C, H, W], arbitrary strides
// Output: out with shape [1, C, H, W]
// For each output element out[0,c,h,w] = (1/B) * sum_{b=0}^{B-1} x[b,c,h,w]
// -----------------------------------------------------------------------------
__global__ void mean_dim0_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    int totalHW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalHW) return;

    int hw = idx;
    int w = hw % W;
    int h = hw / W;

    // Loop over channels
    for (int c = 0; c < C; ++c) {
        const float* base = x + c * sx_c + h * sx_h + w * sx_w;
        float sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            sum += base[b * sx_b];
        }
        out[c * H * W + hw] = sum / (float)B;
    }
}

// -----------------------------------------------------------------------------
// Kernel: group normalization (weight=bias=None, eps=1e-5)
// Input: x with shape [N=1, C, H, W], but we keep general N for flexibility
// Output: y with same shape
// Groups: G = num_groups, channels per group = C/G
// For each group, compute mean and variance over all elements in that group,
// then normalize each element.
// -----------------------------------------------------------------------------
__global__ void group_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int num_groups,
    float eps) {
    int cpg = C / num_groups; // channels per group
    int group_size = N * cpg * H * W; // elements per group
    int group_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (group_idx >= num_groups) return;

    // Shared memory for partial sums within block
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sqsum = smem + blockDim.x;

    int group_start_c = group_idx * cpg;
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    // Each thread processes strided elements in the group (coalesced-ish for contiguous)
    int total_elems = group_size;
    for (int i = tid; i < total_elems; i += blockDim.x) {
        // map i to (n, c_in_group, hw)
        int hw = i % (H * W);
        int c_in_group = (i / (H * W)) % cpg;
        int n = i / (H * W * cpg);
        int c = group_start_c + c_in_group;
        int offset = n * (C * H * W) + c * (H * W) + hw;
        float val = x[offset];
        local_sum += val;
        local_sqsum += val * val;
    }

    s_sum[tid] = local_sum;
    s_sqsum[tid] = local_sqsum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    // Now first thread has total sum and sumsq for this block's portion
    // But we need total over all threads/blocks for this group.
    // We'll use atomicAdd to global accumulators.
    // Since only one block per group is launched, we can just have thread 0 write to global memory.
    // But to be general (if multiple blocks per group), we use atomicAdd.
    // For simplicity and correctness, we launch exactly num_groups blocks (one per group).
    // Then each block computes the whole group reduction and writes mean/var to global.
    // We'll allocate temporary memory for mean and variance in the launcher.
    // The kernel receives pointers to mean and variance arrays (size num_groups).
    if (tid == 0) {
        float mean = s_sum[0] / (float)total_elems;
        float var = s_sqsum[0] / (float)total_elems - mean * mean;
        // Write to global arrays passed as pointers
        // We'll define these as parameters
    }

    // Note: The above is incomplete: we need mean and var passed as pointers.
    // We'll correct the kernel signature and logic below.
}

// We'll define a more complete group_norm kernel using two-pass approach or single-pass with shared mem per block.
// Since group_size may be large (up to 1*8*256*256 = 524288), a single block with 1024 threads and grid-stride loop
// can compute reduction. That's acceptable.

__global__ void group_norm_kernel_v2(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    float* __restrict__ mean_out,
    float* __restrict__ invstd_out) {
    int group_idx = blockIdx.x;
    if (group_idx >= num_groups) return;

    int cpg = C / num_groups;
    int group_size = N * cpg * H * W;
    int totalHW = H * W;

    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sqsum = smem + blockDim.x;

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    int group_start_c = group_idx * cpg;
    // Grid-stride loop over group elements
    for (int i = tid; i < group_size; i += blockDim.x) {
        int hw = i % totalHW;
        int c_in_group = (i / totalHW) % cpg;
        int n = i / (totalHW * cpg);
        int c = group_start_c + c_in_group;
        int offset = n * (C * H * W) + c * totalHW + hw;
        float val = x[offset];
        local_sum += val;
        local_sqsum += val * val;
    }

    s_sum[tid] = local_sum;
    s_sqsum[tid] = local_sqsum;
    __syncthreads();

    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / (float)group_size;
        float var = s_sqsum[0] / (float)group_size - mean * mean;
        mean_out[group_idx] = mean;
        invstd_out[group_idx] = rsqrtf(var + eps);
    }
    __syncthreads();

    // Now normalize and write output
    float mean = mean_out[group_idx];
    float invstd = invstd_out[group_idx];

    for (int i = tid; i < group_size; i += blockDim.x) {
        int hw = i % totalHW;
        int c_in_group = (i / totalHW) % cpg;
        int n = i / (totalHW * cpg);
        int c = group_start_c + c_in_group;
        int offset = n * (C * H * W) + c * totalHW + hw;
        y[offset] = (x[offset] - mean) * invstd;
    }
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C" for binding)
// -----------------------------------------------------------------------------
extern "C" void mean_dim0_keepdim_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    cudaStream_t stream) {
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    int totalHW = H * W;
    int threads = 256;
    int blocks = (totalHW + threads - 1) / threads;
    mean_dim0_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W, sx_b, sx_c, sx_h, sx_w, totalHW);
}

extern "C" void group_norm_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    float* mean_out,
    float* invstd_out,
    cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || num_groups <= 0 || (C % num_groups) != 0) return;

    int threads = 1024;
    int blocks = num_groups;
    size_t shared_mem = 2 * threads * sizeof(float);
    group_norm_kernel_v2<<<blocks, threads, shared_mem, stream>>>(
        x, y, N, C, H, W, num_groups, eps, mean_out, invstd_out);
}
