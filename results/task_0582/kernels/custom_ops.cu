#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper: block reduction for max
__device__ float blockReduceMax(float val, float* shared, int tid, int blockSize) {
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = shared[tid + stride];
            if (other > shared[tid]) shared[tid] = other;
        }
        __syncthreads();
    }
    return shared[0];
}

// Helper: block reduction for sum
__device__ float blockReduceSum(float val, float* shared, int tid, int blockSize) {
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// GroupNorm forward kernel.
// Each block handles one (n, group) pair. threads iterate over (channels_per_group * H * W) elements.
// channel_per_group = C / num_groups. In our contract: C=16, num_groups=16 => channels_per_group=1.
// But keep general: blockDim.x may be 256, elements_per_group may be > blockDim.x, so use grid-stride loop within block.
__global__ void group_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int G,
    float eps
) {
    int group_idx = blockIdx.x;
    int n = group_idx / G;
    int g = group_idx % G;

    int channels_per_group = C / G;
    int group_size = channels_per_group * H * W;

    // Start offset for this (n,g)
    const float* x_group_start = x + (n * C + g * channels_per_group) * H * W;

    // Compute mean and variance
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        float val = x_group_start[idx];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Block reduction for sum and sq_sum
    __shared__ float s_sum[256];
    __shared__ float s_sq_sum[256];
    // blockDim.x <= 256 assumed; but safe with dynamic? We'll use fixed size 256.
    int tid = threadIdx.x;
    if (tid < blockDim.x) {
        s_sum[tid] = local_sum;
        s_sq_sum[tid] = local_sq_sum;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    float sum = s_sum[0];
    float sq_sum = s_sq_sum[0];
    float mean = sum / group_size;
    float var = sq_sum / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Apply normalization and affine
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        // channel index within group: idx / (H*W)
        int c_in_group = idx / (H * W);
        int c_global = g * channels_per_group + c_in_group;
        float val = x_group_start[idx];
        float normalized = (val - mean) * inv_std;
        y[(n * C + c_global) * H * W + (idx % (H * W))] = normalized * weight[c_global] + bias[c_global];
    }
}

// LogSoftmax along dim=1 kernel.
// Each block handles one (n, h, w) position. threads reduce over C.
__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W
) {
    int spatial_idx = blockIdx.x;
    int n = spatial_idx / (H * W);
    int rem = spatial_idx % (H * W);

    const float* x_slice = x + n * C * H * W + rem;
    float* y_slice = y + n * C * H * W + rem;

    // Compute max for numerical stability
    float local_max = -1e30f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = x_slice[c * H * W];
        if (val > local_max) local_max = val;
    }

    __shared__ float s_max[256];
    int tid = threadIdx.x;
    if (tid < blockDim.x) s_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
        }
        __syncthreads();
    }
    float max_val = s_max[0];

    // Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = x_slice[c * H * W];
        local_sum += __expf(val - max_val);
    }

    __shared__ float s_sum[256];
    if (tid < blockDim.x) s_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = s_sum[0];
    float log_sum = logf(sum_exp);

    // Write output: y = x - max - log_sum
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        y_slice[c * H * W] = x_slice[c * H * W] - max_val - log_sum;
    }
}

// Launcher for GroupNorm
extern "C" void group_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N, int C, int H, int W,
    int G,
    float eps,
    cudaStream_t stream
) {
    int total_groups = N * G;
    int threads = 256;
    // grid = total_groups, but if total_groups > INT_MAX? N=512, G=16 => 8192, fine.
    int blocks = total_groups;
    group_norm_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y, N, C, H, W, G, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // We cannot report via exceptions here; just return.
        // Binding layer should check after call.
    }
}

// Launcher for LogSoftmax
extern "C" void log_softmax_dim1_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    int spatial_count = N * H * W;
    int threads = 256;
    // grid = spatial_count, may be large (512*128*128 = 8,388,608 blocks)
    int blocks = spatial_count;
    log_softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore, binding will handle
    }
}
