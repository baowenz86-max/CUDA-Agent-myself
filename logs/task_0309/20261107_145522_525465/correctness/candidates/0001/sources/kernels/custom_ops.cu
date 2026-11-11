#include <cuda_runtime.h>
#include <math.h>

// GroupNorm forward: for each (batch, channel) compute mean and variance over spatial dims,
// then normalize and apply affine (weight, bias).
// Input x: shape [N, C, H, W] as contiguous? Actually we handle arbitrary strides by providing pointers and strides.
// We assume x is a 4D tensor with dimensions N, C, H, W. For generality we could support any spatial dims,
// but given shape [N, C, H, W] we'll implement for 4D. Strides: sx_n, sx_c, sx_h, sx_w from binding as int64_t.
// weight and bias are contiguous arrays of length C (may be null if affine disabled? Contract says nullable but in model they are not None, but handle null).

// Kernel to compute mean and variance for each (batch, channel) using a reduction over H*W elements.
// We launch one block per (batch, channel), blockDim.x = 256 threads, shared memory for reduction.
// Then second kernel to normalize and apply affine using computed means and variances.

// Since num_groups == num_channels in this model, each group is a single channel, but we can implement generalized GroupNorm later if needed.
// For now, given contract, each group has one channel, so we can compute per-channel statistics.
// But to be safe, we can support multiple channels per group by having group_id = channel / channels_per_group.
// However, given num_channels=3, num_groups=3 => channels_per_group=1, group_id = channel. So per-channel is fine.

// We'll write two kernels:
// 1. reduce_statistics_kernel: computes sum and sum_sq per (batch, channel) and stores in buffers mean and var (or sum and sumsq).
// 2. apply_normalization_kernel: uses mean and var to compute output.
// Or we can do it in one kernel using a two-pass approach per block: block computes mean/var for its assigned (batch, channel) and then applies normalization to all elements in that (batch, channel) potentially relying on global sync? Not possible across blocks. So separate kernels are needed.

// Alternative: one kernel that computes mean/var per (batch, channel) into global memory, then second kernel for normalization.
// We'll do that with two kernels and an intermediate buffer allocated by binding layer (we'll assume binding provides mean and var arrays of size N*C).
// Actually we can let the launcher allocate with cudaMalloc? Binding should allocate using torch empty and pass pointers.
// Since .cu cannot use torch, we'll declare launcher that takes output, x, weight, bias, mean_buf, var_buf, N, C, H, W, strides, eps, stream.
// Binding will allocate mean_buf and var_buf with torch::empty({N*C}, options).

// Kernel 1: block per (n,c), threads compute partial sum and sum_sq using grid-stride loop over spatial positions.
// Use shared memory to reduce within block, then one thread writes mean and var (or sum and sumsq) to global.

// Kernel 2: block per (n,c) or grid-stride over all elements, each thread computes normalized value using mean[n*C+c], var[n*C+c].
// For better performance, we can fuse into one kernel using one block per (n,c) and two phases with __syncthreads inside, but need to write to global and sync across blocks impossible. So two kernels.

// We'll use float for all intermediate calculations.

// Define block size for reduction
#define REDUCE_BLOCK_SIZE 256

__global__ void compute_mean_var_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ var,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w)
{
    int block_id = blockIdx.x; // corresponds to (n, c) flattened index
    int idx = block_id; // block_id = n * C + c
    // Actually gridDim.x = N*C, blockIdx.x = n*C + c
    int c = idx % C;
    int n = idx / C;
    int total_spatial = H * W;

    const float* base_x = x + n * stride_n + c * stride_c;

    __shared__ float s_sum[REDUCE_BLOCK_SIZE];
    __shared__ float s_sumsq[REDUCE_BLOCK_SIZE];

    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    // Grid-stride loop over spatial positions within this (n,c)
    for (int i = tid; i < total_spatial; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        float val = base_x[h * stride_h + w * stride_w]; // assuming stride_w=1? Actually stride_w is given, could be non-contiguous.
        local_sum += val;
        local_sumsq += val * val;
    }

    // Reduce within block
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();

    // Tree reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sumsq[tid] += s_sumsq[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float count = (float)total_spatial;
        float mean_val = s_sum[0] / count;
        float var_val = s_sumsq[0] / count - mean_val * mean_val;
        mean[idx] = mean_val;
        var[idx] = var_val;
    }
}

__global__ void normalize_apply_affine_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    float eps)
{
    int total = N * C * H * W;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear_idx = tid; linear_idx < total; linear_idx += stride) {
        // Decode linear index to n,c,h,w
        int w = linear_idx % W;
        int tmp = linear_idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        int idx_stat = n * C + c;
        float m = mean[idx_stat];
        float v = var[idx_stat];
        float inv_std = rsqrtf(v + eps);
        float val = x[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
        float normalized = (val - m) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[c];
        }
        if (bias != nullptr) {
            normalized += bias[c];
        }
        out[n * stride_n + c * stride_c + h * stride_h + w * stride_w] = normalized;
    }
}

// Softmax over channel dimension for each (batch, spatial location).
// Input x shape [N, C, H, W], output same shape. Compute max over C for each n,h,w, then exp and sum, then output exp(x - max)/sum.
// Since C=3 small, we allocate one thread per (n,h,w) and loop over C.
__global__ void softmax_channel_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w)
{
    int total_spatial = N * H * W;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_spatial; idx += stride) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        int n = tmp / H;

        // Compute max over channels
        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = x[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
            if (val > max_val) max_val = val;
        }

        // Compute sum of exp
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = x[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
            sum_exp += expf(val - max_val);
        }

        // Write output
        for (int c = 0; c < C; ++c) {
            float val = x[n * stride_n + c * stride_c + h * stride_h + w * stride_w];
            float softmax_val = expf(val - max_val) / sum_exp;
            out[n * stride_n + c * stride_c + h * stride_h + w * stride_w] = softmax_val;
        }
    }
}

// Launcher for group_norm_forward
// Arguments:
//   out: output tensor (same shape as x)
//   x: input tensor
//   weight: affine weight (length C), can be null
//   bias: affine bias (length C), can be null
//   mean_buf, var_buf: temporary buffers of size N*C allocated by binding (float)
//   N, C, H, W: dimensions
//   stride_n, stride_c, stride_h, stride_w: strides of x (and out, assumed same as x? Contract says output same shape and layout as input, so we can use same strides for out; but binding should ensure out has same stride as x if using torch::empty_like, which preserves stride. We'll assume out uses the same strides as x, passed as parameters)
//   eps
//   stream
extern "C" void group_norm_forward_launcher(
    float* out,
    const float* x,
    const float* weight,
    const float* bias,
    float* mean_buf,
    float* var_buf,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    float eps,
    cudaStream_t stream)
{
    // Compute mean and variance
    int num_stats = N * C;
    dim3 grid_stats(num_stats);
    dim3 block_stats(REDUCE_BLOCK_SIZE);
    compute_mean_var_kernel<<<grid_stats, block_stats, 0, stream>>>(
        x, mean_buf, var_buf, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w);

    // Normalize and apply affine
    int total_elements = N * C * H * W;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    if (num_blocks > 65535) num_blocks = 65535; // grid-stride loop handles rest
    dim3 grid_norm(num_blocks);
    dim3 block_norm(block_size);
    normalize_apply_affine_kernel<<<grid_norm, block_norm, 0, stream>>>(
        x, mean_buf, var_buf, out, weight, bias, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w, eps);
}

// Launcher for softmax_channel_forward
extern "C" void softmax_channel_forward_launcher(
    float* out,
    const float* x,
    int N, int C, int H, int W,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    int total_spatial = N * H * W;
    int block_size = 256;
    int num_blocks = (total_spatial + block_size - 1) / block_size;
    if (num_blocks > 65535) num_blocks = 65535; // grid-stride loop handles rest
    dim3 grid(num_blocks);
    dim3 block(block_size);
    softmax_channel_kernel<<<grid, block, 0, stream>>>(
        x, out, N, C, H, W, stride_n, stride_c, stride_h, stride_w);
}
