#include <cuda_runtime.h>

// GroupNorm forward kernel: first stage computes per (n,c) sum and sum_sq
__global__ void group_norm_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t group_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C;
    if (idx >= total) return;

    int64_t n = idx / C;
    int64_t c = idx % C;

    const float* x = input + n * C * H * W + c * H * W;
    float s = 0.f;
    float ss = 0.f;
    for (int64_t i = 0; i < H * W; ++i) {
        float v = x[i];
        s += v;
        ss += v * v;
    }
    sum[idx] = s;
    sum_sq[idx] = ss;
}

// Second stage: compute per (n, group) mean and inv_std
__global__ void group_norm_stats_kernel(
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int64_t N, int64_t num_groups, int64_t group_size,
    double eps) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * num_groups;
    if (idx >= total) return;

    int64_t n = idx / num_groups;
    int64_t g = idx % num_groups;

    int64_t group_channels = group_size;
    int64_t start_c = g * group_channels;
    int64_t end_c = start_c + group_channels;

    float s = 0.f;
    float ss = 0.f;
    for (int64_t c = start_c; c < end_c; ++c) {
        int64_t sc_idx = n * (num_groups * group_channels) + c;
        s += sum[sc_idx];
        ss += sum_sq[sc_idx];
    }

    int64_t count = group_channels * H * W;  // H*W not passed, but we can recompute? We don't have H,W here. Pass count explicitly.
    // Need H,W, so H*W must be passed as count
    // We'll instead pass spatial_size (H*W) to this kernel as well.
    // Modify launcher accordingly.
}

// We'll define a combined kernel that avoids multiple kernels: one kernel per (n,c) computes stats and then group stats, but that requires synchronization across group. Simpler: pass spatial size.
// Revised:

__global__ void group_norm_stats_kernel(
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int64_t N, int64_t C, int64_t num_groups, int64_t group_size, int64_t spatial_size,
    double eps) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * num_groups;
    if (idx >= total) return;

    int64_t n = idx / num_groups;
    int64_t g = idx % num_groups;

    int64_t group_channels = group_size;
    int64_t start_c = g * group_channels;
    int64_t end_c = start_c + group_channels;

    float s = 0.f;
    float ss = 0.f;
    for (int64_t c = start_c; c < end_c; ++c) {
        int64_t sc_idx = n * C + c;
        s += sum[sc_idx];
        ss += sum_sq[sc_idx];
    }

    int64_t count = group_channels * spatial_size;
    float mu = s / count;
    float var = ss / count - mu * mu;
    float inv_std = rsqrtf(var + (float)eps);
    mean[idx] = mu;
    inv_std[idx] = inv_std;
}

// Final output kernel: apply normalization and affine
__global__ void group_norm_apply_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t num_groups, int64_t group_size) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H * W;
    if (idx >= total) return;

    int64_t spatial_idx = idx % (H * W);
    int64_t rem = idx / (H * W);
    int64_t c = rem % C;
    int64_t n = rem / C;

    int64_t g = c / group_size;
    int64_t stat_idx = n * num_groups + g;
    float mu = mean[stat_idx];
    float inv_std_val = inv_std[stat_idx];
    float x = input[idx];
    float normalized = (x - mu) * inv_std_val;
    output[idx] = normalized * weight[c] + bias[c];
}

// Launcher
extern "C" void group_norm_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    double eps,
    cudaStream_t stream) {

    int64_t spatial_size = H * W;
    int64_t group_size = C / num_groups;
    int64_t NC = N * C;
    int64_t NG = N * num_groups;
    int64_t total = N * C * spatial_size;

    // Temporary buffers for sum, sum_sq, mean, inv_std
    float *d_sum, *d_sum_sq, *d_mean, *d_inv_std;
    cudaMalloc(&d_sum, NC * sizeof(float));
    cudaMalloc(&d_sum_sq, NC * sizeof(float));
    cudaMalloc(&d_mean, NG * sizeof(float));
    cudaMalloc(&d_inv_std, NG * sizeof(float));

    int threads = 256;
    int blocks_sum = (NC + threads - 1) / threads;
    int blocks_stats = (NG + threads - 1) / threads;
    int blocks_apply = (total + threads - 1) / threads;

    group_norm_sum_kernel<<<blocks_sum, threads, 0, stream>>>(
        input, d_sum, d_sum_sq, N, C, H, W, group_size);

    group_norm_stats_kernel<<<blocks_stats, threads, 0, stream>>>(
        d_sum, d_sum_sq, d_mean, d_inv_std, N, C, num_groups, group_size, spatial_size, eps);

    group_norm_apply_kernel<<<blocks_apply, threads, 0, stream>>>(
        input, d_mean, d_inv_std, weight, bias, output, N, C, H, W, num_groups, group_size);

    cudaFree(d_sum);
    cudaFree(d_sum_sq);
    cudaFree(d_mean);
    cudaFree(d_inv_std);
}