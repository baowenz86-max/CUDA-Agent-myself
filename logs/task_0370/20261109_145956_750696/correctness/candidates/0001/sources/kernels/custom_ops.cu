#include <cuda_runtime.h>

// GroupNorm forward kernels.
// Input layout: [N, C, H, W] with arbitrary strides? We assume contiguous for simplicity
// because the binding will call .contiguous() if needed. The contract says "supports strides"
// but we can handle by using input pointer directly assuming contiguous; in the binding we will
// enforce contiguity. To be safe, the launcher accepts raw pointers and assumes contiguous.

// Reduction kernel: compute mean and variance for each (n, group)
// Each block handles one (n, group). Block size = 256 up to 1024.
// We use a grid-stride loop within the block over all elements in the group.
// Dimensions: input shape [N, C, H, W], group_size = (C / num_groups) * H * W.
// channel_offset = g * (C / num_groups)
// The group contains channels [g * channels_per_group, (g+1)*channels_per_group)
__global__ void group_norm_mean_var_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    int64_t channels_per_group)
{
    // blockIdx.x = index = n * num_groups + g
    int64_t index = blockIdx.x;
    int64_t n = index / num_groups;
    int64_t g = index % num_groups;
    int64_t group_size = channels_per_group * H * W;
    // Base pointer for this group: input + n*C*H*W + g*channels_per_group*H*W
    const float* group_base = input + n * C * H * W + g * channels_per_group * H * W;

    // Each thread computes partial sum and sum of squares over its strided elements.
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = group_base[i];
        sum += val;
        sumsq += val * val;
    }

    // Reduction within block using shared memory.
    __shared__ float s_sum[256];  // blockDim.x max 256? We'll use fixed 256 threads.
    __shared__ float s_sumsq[256];
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = s_sum[0] / group_size;
        float var = s_sumsq[0] / group_size - mean * mean;
        // PyTorch uses biased variance. Ensure non-negative due to float.
        if (var < 0.0f) var = 0.0f;
        mean_out[index] = mean;
        var_out[index] = var;
    }
}

// Normalization kernel: output = (input - mean) / sqrt(var + eps) * weight + bias
__global__ void group_norm_apply_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    int64_t channels_per_group,
    double eps)
{
    int64_t total_elements = N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decode indices
    int64_t w = idx % W;
    int64_t tmp = idx / W;
    int64_t h = tmp % H;
    tmp = tmp / H;
    int64_t c = tmp % C;
    int64_t n = tmp / C;

    int64_t g = c / channels_per_group;
    int64_t group_index = n * num_groups + g;
    float m = mean[group_index];
    float v = var[group_index];
    float inv_std = rsqrtf(v + (float)eps);
    float normalized = (input[idx] - m) * inv_std;
    float w_val = (weight != nullptr) ? weight[c] : 1.0f;
    float b_val = (bias != nullptr) ? bias[c] : 0.0f;
    output[idx] = normalized * w_val + b_val;
}

// Host launcher (extern "C")
extern "C" void group_norm_forward_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    double eps,
    cudaStream_t stream)
{
    int64_t channels_per_group = C / num_groups;
    int64_t num_groups_total = N * num_groups;

    // Allocate temporary storage for mean and var
    float* d_mean = nullptr;
    float* d_var = nullptr;
    cudaMalloc(&d_mean, num_groups_total * sizeof(float));
    cudaMalloc(&d_var, num_groups_total * sizeof(float));

    // Reduction kernel: one block per (n, group), fixed block size 256 (max 1024)
    int block_size = 256;
    // Ensure block size not exceed max threads (1024)
    if (block_size > 1024) block_size = 1024;
    dim3 grid(num_groups_total);
    group_norm_mean_var_kernel<<<grid, block_size, 0, stream>>>(
        input, d_mean, d_var, N, C, H, W, num_groups, channels_per_group);

    // Normalization kernel: grid-stride or exact?
    int64_t total_elements = N * C * H * W;
    int apply_block_size = 256;
    int64_t apply_grid = (total_elements + apply_block_size - 1) / apply_block_size;
    group_norm_apply_kernel<<<apply_grid, apply_block_size, 0, stream>>>(
        input, output, d_mean, d_var, weight, bias, N, C, H, W, num_groups, channels_per_group, eps);

    // Free temporary memory asynchronously? We should synchronize before freeing to ensure kernels done.
    cudaStreamSynchronize(stream);
    cudaFree(d_mean);
    cudaFree(d_var);
}

// Note: The binding layer will call cudaGetLastError() after the launcher returns, as kernels are async.
