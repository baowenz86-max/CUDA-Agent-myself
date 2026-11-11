#include <cuda_runtime.h>
#include <math.h>

// Forward declarations for helper functions, or reorder definitions.
// Define warpReduceSum first.
__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float blockReduceSum(float val) {
    // Shared memory for partial sums
    __shared__ float shared[32]; // max block size 1024, but we'll use up to 1024; this assumes blockDim.x <= 1024 and warpSize=32
    int lane = threadIdx.x & 31; // threadIdx.x % warpSize
    int wid = threadIdx.x >> 5; // threadIdx.x / warpSize

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared if needed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;

    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Kernel for group normalization.
// Each block processes one group of one batch element.
// x: input, shape [N, C, L], float32, may be non-contiguous.
// y: output, same shape.
// N: batch size, C: channels, L: spatial length.
// groups: number of groups, G = groups, channels_per_group = C / groups.
// eps: epsilon for numerical stability (default 1e-5).
// We assume x and y are contiguous in the same manner? Actually, binding should ensure they are either contiguous or pass strides.
// For simplicity, this kernel assumes contiguous tensors. Binding code should make x contiguous if not, or pass strides and adjust indexing.
// Here we only implement contiguous case because model inputs are contiguous.
__global__ void group_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N,
    int C,
    int L,
    int groups,
    float eps)
{
    int group_size = (C / groups) * L; // number of elements per group per batch
    int batch = blockIdx.x;            // batch index
    int group = blockIdx.y;            // group index within batch
    int tid = threadIdx.x;

    int channels_per_group = C / groups;
    int group_start_channel = group * channels_per_group;

    // Shared memory for mean and variance computation
    // We use static shared memory inside blockReduceSum, so no extern needed.
    __shared__ float s_mean;
    __shared__ float s_var;

    // Step 1: compute mean using parallel reduction
    float sum = 0.0f;
    int spatial_offset = batch * C * L + group_start_channel * L;
    for (int i = tid; i < group_size; i += blockDim.x) {
        // i ranges over elements within group: channel_offset * L + spatial_idx
        int channel_in_group = i / L;
        int spatial_idx = i % L;
        int idx = spatial_offset + channel_in_group * L + spatial_idx;
        sum += x[idx];
    }

    // Warp reduction within block
    sum = blockReduceSum(sum);

    if (tid == 0) {
        s_mean = sum / group_size;
    }
    __syncthreads();

    float mean = s_mean;

    // Step 2: compute variance
    float sq_sum = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int channel_in_group = i / L;
        int spatial_idx = i % L;
        int idx = spatial_offset + channel_in_group * L + spatial_idx;
        float diff = x[idx] - mean;
        sq_sum += diff * diff;
    }

    sq_sum = blockReduceSum(sq_sum);

    if (tid == 0) {
        s_var = sq_sum / group_size; // biased variance as in PyTorch
    }
    __syncthreads();

    float var = s_var;
    float inv_std = rsqrtf(var + eps);

    // Step 3: normalize and write output
    for (int i = tid; i < group_size; i += blockDim.x) {
        int channel_in_group = i / L;
        int spatial_idx = i % L;
        int idx = spatial_offset + channel_in_group * L + spatial_idx;
        y[idx] = (x[idx] - mean) * inv_std;
    }
}

// Launcher for group norm. Assumes x and y are contiguous float32 CUDA tensors.
// Binding code must pass consistent strides or ensure contiguity.
extern "C" void custom_group_norm_launcher(
    const float* x,
    float* y,
    int N,
    int C,
    int L,
    int groups,
    float eps,
    cudaStream_t stream)
{
    int group_size = (C / groups) * L;
    // Choose block size: cap at 512 threads
    int block_size = 256;
    if (group_size < 256) block_size = 32; // minimum
    if (group_size < 32) block_size = group_size; // handle very small groups, but ensure at least 1
    if (block_size < 1) block_size = 1;

    dim3 grid(N, groups);
    int shared_mem = 0; // we don't actually use dynamic shared memory, but blockReduceSum uses static shared
    // Actually we need dynamic shared for blockReduceSum; but we can use static __shared__ in the helper, which is per-block.
    // So no dynamic shared memory needed.

    group_norm_kernel<<<grid, block_size, 0, stream>>>(
        x, y, N, C, L, groups, eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // We cannot report via exception here; binding code will check after launch.
    }
}

// Kernel for 1D transposed convolution (also called deconvolution).
// Assumes inputs and weight are contiguous float32.
// x: input [N, C_in, L_in] (after possible contiguous conversion)
// w: weight [C_in, C_out, K]
// bias: pointer to bias [C_out] or nullptr if no bias
// y: output [N, C_out, L_out]
// L_out = (L_in - 1) * stride - 2 * padding + K + output_padding
// This kernel uses one thread per output element.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    bool has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int ol = idx % L_out;
    int tmp = idx / L_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        const float* x_ptr = x + n * C_in * L_in + ic * L_in;
        const float* w_ptr = w + ic * C_out * K + oc * K;
        for (int k = 0; k < K; ++k) {
            int il = ol + padding - k;
            if (il % stride != 0) continue;
            il /= stride;
            if (il >= 0 && il < L_in) {
                acc += x_ptr[il] * w_ptr[k];
            }
        }
    }
    if (has_bias) {
        acc += bias[oc];
    }
    y[idx] = acc;
}

// Launcher for transposed convolution. Raw pointers, all tensors assumed contiguous.
// x, w, bias (can be nullptr), y are device pointers.
extern "C" void custom_conv_transpose1d_launcher(
    const float* x,
    const float* w,
    const float* bias,
    float* y,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    bool has_bias,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    conv_transpose1d_kernel<<<grid_size, block_size, 0, stream>>>(
        x, w, bias, y,
        N, C_in, C_out, L_in, L_out,
        K, stride, padding, output_padding, has_bias);
}
