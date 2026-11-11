#include <cuda_runtime.h>
#include <cstdint>

// Helper: ceil division
static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// GroupNorm kernel
// Each block handles one (batch, group) pair and uses block reduction via shared memory.
// Input layout: [B, C, H, W] with C channels, each group has group_size channels.
// channel_stride = H*W, spatial_size = H*W, group_spatial = group_size*spatial_size
// Output has same layout.
__global__ void group_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int B,
    int C,
    int H,
    int W,
    int num_groups,
    float eps
) {
    int group_size = C / num_groups; // channels per group
    int spatial_size = H * W;
    int group_spatial = group_size * spatial_size; // elements per group per batch
    
    int group_per_batch = num_groups;
    int total_groups = B * num_groups;
    
    for (int group_idx = blockIdx.x; group_idx < total_groups; group_idx += gridDim.x) {
        int b = group_idx / num_groups;
        int g = group_idx % num_groups;
        const float* x_group = x + b * C * spatial_size + g * group_spatial;
        float* y_group = y + b * C * spatial_size + g * group_spatial;
        
        // Compute mean and variance using block reduction
        extern __shared__ float smem[]; // two arrays: mean and variance
        float* s_mean = smem;
        float* s_var = &smem[blockDim.x];
        
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int idx = threadIdx.x;
        int stride = blockDim.x;
        for (int i = idx; i < group_spatial; i += stride) {
            float val = x_group[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // Block reduction for sum and sum_sq
        // Use warp shuffle then shared memory for cross-warp reduction
        // Simple approach: shared memory reduction with __syncthreads
        s_mean[idx] = sum;
        s_var[idx] = sum_sq;
        __syncthreads();
        
        // Reduce within block using shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (idx < s) {
                s_mean[idx] += s_mean[idx + s];
                s_var[idx] += s_var[idx + s];
            }
            __syncthreads();
        }
        
        float total_sum = s_mean[0];
        float total_sum_sq = s_var[0];
        float mean = total_sum / group_spatial;
        float var = total_sum_sq / group_spatial - mean * mean;
        float inv_std = rsqrtf(var + eps);
        
        // Apply normalization and affine
        for (int i = idx; i < group_spatial; i += stride) {
            int local_channel = i / spatial_size;
            int pos = i % spatial_size;
            int global_channel = g * group_size + local_channel;
            float normalized = (x_group[i] - mean) * inv_std;
            if (weight != nullptr) normalized *= weight[global_channel];
            if (bias != nullptr) normalized += bias[global_channel];
            y_group[i] = normalized;
        }
    }
}

// PermuteSum kernel: sum over C dimension, output shape [B,H,W,1]
// Output stride: [H*W, W, 1, H*W] as per reference (but actual memory layout is contiguous with shape [B,H,W,1])
// We allocate contiguous output and write accordingly: output linear index = ((b*H + h)*W + w)*1
// Input linear index = b*C*H*W + c*H*W + h*W + w
__global__ void permute_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B,
    int C,
    int H,
    int W
) {
    int total = B * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int b = tmp / H;
    
    float sum = 0.0f;
    const float* base = x + b * C * H * W + h * W + w;
    for (int c = 0; c < C; ++c) {
        sum += base[c * H * W];
    }
    out[idx] = sum;
}

// Launcher for custom_group_norm
extern "C" void custom_group_norm_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int B,
    int C,
    int H,
    int W,
    int num_groups,
    float eps,
    cudaStream_t stream
) {
    int group_size = C / num_groups;
    int spatial_size = H * W;
    int group_spatial = group_size * spatial_size;
    
    // Choose block size: use 256 threads; if group_spatial smaller, use smaller block.
    int threads = std::min(256, group_spatial);
    if (threads == 0) threads = 1;
    // Ensure threads is power of two for reduction
    int power = 1;
    while (power * 2 <= threads) power *= 2;
    threads = power;
    
    int groups_total = B * num_groups;
    // Use enough blocks to cover all groups, but also grid-stride loop in kernel handles multiple groups per block if needed.
    int blocks = std::min(groups_total, 65535); // gridDim.x max 65535 in X dimension
    if (blocks == 0) blocks = 1;
    
    size_t smem_bytes = 2 * threads * sizeof(float);
    
    group_norm_kernel<<<blocks, threads, smem_bytes, stream>>>(
        x, y, weight, bias, B, C, H, W, num_groups, eps
    );
}

// Launcher for custom_permute_sum
extern "C" void custom_permute_sum_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream
) {
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_sum_kernel<<<blocks, threads, 0, stream>>>(x, out, B, C, H, W);
}
