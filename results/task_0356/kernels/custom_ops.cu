#include <cuda_runtime.h>
#include <cmath>

// ----------------------------------------------------------------------------
// BatchNorm2d evaluation mode kernel
// y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Input x is 4D (N, C, H, W) with arbitrary strides.
// weight, bias, running_mean, running_var are contiguous 1D of length C.
// ----------------------------------------------------------------------------
__global__ void batch_norm_eval_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long total = (long long)N * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        float x = input[n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float normalized = (x - mean) * inv_std;
        float y = normalized * weight[c] + bias[c];

        output[n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w] = y;
    }
}

// ----------------------------------------------------------------------------
// GroupNorm forward: computes mean and variance per (n, group), then normalizes.
// Input x is 4D (N, C, H, W) with arbitrary strides.
// weight, bias are contiguous 1D of length C.
// num_groups divides C; channels_per_group = C / num_groups.
// ----------------------------------------------------------------------------

// Kernel to compute partial sums for mean and variance per (n, group).
// Each block handles one (n, group) pair; blockDim.x threads process the group's
// channels * H * W elements using grid-stride within the block (loop).
// We use a two-pass approach: first kernel computes sum and sum of squares,
// second kernel normalizes using global mean and variance.
__global__ void group_norm_stats_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int H, int W, int num_groups,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    float* __restrict__ mean_out,
    float* __restrict__ var_out)
{
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sqsum = smem + blockDim.x;

    int group = blockIdx.x;
    int n = blockIdx.y; // gridDim.y = N
    int channels_per_group = C / num_groups;
    int group_start_c = group * channels_per_group;
    int group_end_c = group_start_c + channels_per_group;
    int count = channels_per_group * H * W;

    // Initialize shared memory
    s_sum[threadIdx.x] = 0.0f;
    s_sqsum[threadIdx.x] = 0.0f;
    __syncthreads();

    // Grid-stride loop over group elements
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int c_offset = i / (H * W);
        int spatial = i % (H * W);
        int h = spatial / W;
        int w = spatial % W;
        int c = group_start_c + c_offset;
        long long idx = (long long)n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        float val = input[idx];
        s_sum[threadIdx.x] += val;
        s_sqsum[threadIdx.x] += val * val;
    }
    __syncthreads();

    // Reduce within block using warp shuffle and shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
            s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = s_sum[0];
        float sqsum = s_sqsum[0];
        float mean = sum / count;
        float var = sqsum / count - mean * mean;
        // Use max(var, 0) to avoid negative due to floating point
        var = fmaxf(var, 0.0f);
        int index = n * num_groups + group;
        mean_out[index] = mean;
        var_out[index] = var;
    }
}

// Kernel to normalize and apply affine transformation
__global__ void group_norm_apply_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int N, int C, int H, int W, int num_groups,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long total = (long long)N * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        int channels_per_group = C / num_groups;
        int group = c / channels_per_group;
        int index = n * num_groups + group;
        float m = mean[index];
        float v = var[index];
        float inv_std = rsqrtf(v + eps);
        float normalized = (input[n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w] - m) * inv_std;
        float y = normalized * weight[c] + bias[c];
        output[n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w] = y;
    }
}

// ----------------------------------------------------------------------------
// Host launchers (C interface, no PyTorch dependencies)
// These are called from the binding .cpp with raw pointers, dimensions, strides,
// and cudaStream_t.
// ----------------------------------------------------------------------------

extern "C" void batch_norm_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int block = 256;
    int grid = (int)min((total + block - 1) / block, (long long)2147483647);
    if (grid == 0) grid = 1;
    batch_norm_eval_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

extern "C" void group_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float eps,
    int N, int C, int H, int W, int num_groups,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float* mean_buffer,
    float* var_buffer,
    cudaStream_t stream)
{
    // Stats kernel: grid.y = N, grid.x = num_groups
    dim3 grid_stats(num_groups, N);
    int block_stats = 256;
    // Shared memory for two float arrays of size block_stats
    size_t shared_mem = 2 * block_stats * sizeof(float);
    group_norm_stats_kernel<<<grid_stats, block_stats, shared_mem, stream>>>(
        input, weight, bias, eps,
        N, C, H, W, num_groups,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        mean_buffer, var_buffer);

    // Apply kernel: 1D grid over all elements
    long long total = (long long)N * C * H * W;
    int block_apply = 256;
    int grid_apply = (int)min((total + block_apply - 1) / block_apply, (long long)2147483647);
    if (grid_apply == 0) grid_apply = 1;
    group_norm_apply_kernel<<<grid_apply, block_apply, 0, stream>>>(
        output, input, weight, bias, mean_buffer, var_buffer, eps,
        N, C, H, W, num_groups,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
