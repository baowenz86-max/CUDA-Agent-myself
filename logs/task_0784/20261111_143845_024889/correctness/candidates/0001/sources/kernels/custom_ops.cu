#include <cuda_runtime.h>
#include <cstdint>

// GroupNorm specialized for num_groups == num_channels (per-channel normalization)
// Input: x shape [N, C, H, W] (4D, contiguous or non-contiguous)
// weight, bias: optional per-channel affine parameters, length C
// eps: epsilon for numerical stability
// Output: same shape as x

// Block size for reduction kernels
constexpr int REDUCE_BLOCK_SIZE = 256;

// Kernel to compute per-(n,c) mean and variance over H*W spatial elements
// Since num_groups == C, each group is one channel.
// We launch grid of (N*C) blocks, each block reduces one (n,c) slice.
__global__ void group_norm_mean_var_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int H, int W,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    int spatial_size)
{
    int idx = blockIdx.x; // index in [0, N*C)
    if (idx >= N * C) return;
    int n = idx / C;
    int c = idx - n * C;

    const float* x_slice = x + n * stride_n + c * stride_c;

    // Each thread accumulates partial sum and sum of squares using grid-stride loop over spatial_size
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int h = i / W;
        int w = i - h * W;
        float val = x_slice[h * stride_h + w * stride_w];
        sum += val;
        sq_sum += val * val;
    }

    // Block reduction for sum and sq_sum
    __shared__ float s_sum[REDUCE_BLOCK_SIZE];
    __shared__ float s_sq_sum[REDUCE_BLOCK_SIZE];

    s_sum[threadIdx.x] = sum;
    s_sq_sum[threadIdx.x] = sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_sum = s_sum[0];
        float total_sq_sum = s_sq_sum[0];
        float mean_val = total_sum / spatial_size;
        float var_val = fmaxf(total_sq_sum / spatial_size - mean_val * mean_val, 0.0f);
        mean[idx] = mean_val;
        var[idx] = var_val;
    }
}

// Kernel to apply normalization and affine: y = (x - mean) / sqrt(var + eps) * weight + bias
// Launch one thread per output element with grid-stride loop over total elements
__global__ void group_norm_apply_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int H, int W,
    int64_t stride_n_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    int64_t total_elements)
{
    int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t idx = global_idx; idx < total_elements; idx += stride) {
        // Decompose linear index into n,c,h,w assuming output is contiguous? 
        // But output might be non-contiguous? Contract says output same shape, but we can assume contiguous for simplicity.
        // For safety, we use the output strides provided. We assume output layout same as input? 
        // Contract example input is contiguous, output stride matches input. We'll compute using linear index over contiguous memory
        // of output, but we don't know if output is contiguous. Better: use input strides for x, but output y should be written at
        // same strides as input? The binding will allocate output with same shape and likely contiguous (PyTorch empty_like preserves strides?).
        // Actually torch::empty_like(x) will allocate contiguous tensor with same shape but default strides, which may differ if x is non-contiguous.
        // To handle general case, pass output strides as well. We'll assume output is contiguous with shape [N,C,H,W] because PyTorch empty_like for non-contiguous returns contiguous tensor of same shape.
        // However, contract says output stride matches input stride. In Python, output from cuda_extension likely created via torch.empty_like(x) which for contiguous input gives contiguous output, matching strides.
        // We'll implement using linear indexing assuming output is contiguous, but provide output strides for safety.
        // For simplicity, we assume x and y are both contiguous (which is true for the given task). We'll still pass strides but use them for x only.
        // Actually we need to ensure correct performance: total_elements = N*C*H*W, and we want to write y[idx] = ... for contiguous y.
        // Use idx to compute n,c,h,w from output strides if possible; but we can compute from linear index assuming contiguous output (row-major).
        // In PyTorch, empty_like creates contiguous tensor, so output strides are computed as [C*H*W, H*W, W, 1]. We'll use those.
        int n = idx / (C * H * W);
        int rem = idx - n * (C * H * W);
        int c = rem / (H * W);
        int rem2 = rem - c * (H * W);
        int h = rem2 / W;
        int w = rem2 - h * W;

        // Map to input using input strides
        const float* x_ptr = x + n * stride_n_in + c * stride_c_in + h * stride_h_in + w * stride_w_in;
        float* y_ptr = y + n * stride_n_out + c * stride_c_out + h * stride_h_out + w * stride_w_out;

        int nc = n * C + c;
        float mean_val = mean[nc];
        float var_val = var[nc];
        float inv_std = rsqrtf(var_val + eps);
        float val = (*x_ptr - mean_val) * inv_std;
        if (weight != nullptr) val *= weight[c];
        if (bias != nullptr) val += bias[c];
        *y_ptr = val;
    }
}

// Launcher function (C interface)
extern "C" void group_norm_forward_launcher(
    const float* x,           // input pointer
    float* y,                 // output pointer
    const float* weight,      // may be null
    const float* bias,        // may be null
    int N, int C, int H, int W,
    int64_t stride_n_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    int num_groups,
    double eps_double,
    cudaStream_t stream)
{
    // Since num_groups == C assumed, but we allow general? We'll implement assuming num_groups divides C and groups are contiguous channels,
    // but for num_groups == C it's per-channel. In general, group size = C / num_groups. This launcher will support arbitrary num_groups.
    // We'll adjust: each group contains group_size channels, and we compute mean/var for each (n, group) over all channels in group and all spatial positions.
    // However, our current kernel assumes per (n,c) reduction. Need to modify to support groups.
    // For simplicity and because contract says num_groups=3, C=3, we implement only per-channel. But we should make it correct for general case?
    // The contract explicitly says "num_groups must divide C (here C=3 and num_groups=3, so one group per channel)". So it's safe to assume group_size=1.
    // We'll implement for arbitrary num_groups by aggregating over group channels. We'll modify kernel accordingly.
    // For now, we'll implement per-channel and assume num_groups == C. If not, we could fallback? But binding will check.
    // We'll implement general to be safe, but it's more complex. We'll stay with per-channel since that's the only exercised case.
    // The contract says "only implement dtypes/devices exercised by this task unless extra tests are supplied." So per-channel is fine.

    int spatial_size = H * W;
    int num_channels = C;
    // We'll allocate mean and var arrays on device for N*C entries
    // Use cudaMalloc and cudaFree inside launcher
    float *d_mean, *d_var;
    cudaMalloc(&d_mean, N * C * sizeof(float));
    cudaMalloc(&d_var, N * C * sizeof(float));

    // Launch reduction kernel: one block per (n,c)
    int num_blocks = N * C;
    int block_size = REDUCE_BLOCK_SIZE;
    group_norm_mean_var_kernel<<<num_blocks, block_size, 0, stream>>>(
        x, d_mean, d_var,
        N, C, H, W,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        spatial_size);

    // Launch apply kernel over entire output
    int64_t total_elements = (int64_t)N * C * H * W;
    int apply_block_size = 256;
    int grid_size = (int)((total_elements + apply_block_size - 1) / apply_block_size);
    // Ensure grid doesn't exceed max grid dimension (2^31-1), but for our size it's fine
    group_norm_apply_kernel<<<grid_size, apply_block_size, 0, stream>>>(
        x, y, d_mean, d_var, weight, bias,
        (float)eps_double,
        N, C, H, W,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out,
        total_elements);

    // Free temporary memory
    cudaFree(d_mean);
    cudaFree(d_var);
}
