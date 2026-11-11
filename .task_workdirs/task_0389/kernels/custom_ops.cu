#include <cuda_runtime.h>
#include <cmath>

// CUDA kernels for batch_norm2d_eval and softmax_channel
// All tensors are float32 on CUDA with shape [N, C, H, W]
// Strides support non-contiguous inputs

// Kernel: batch norm inference
// y[n,c,h,w] = (x[n,c,h,w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// If weight or bias is null, omit corresponding term.
__global__ void batch_norm2d_eval_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,  // may be null
    const float* __restrict__ bias,    // may be null
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    double eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w)
{
    int64_t total = N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        // decompose i into n, c, h, w
        int64_t w = i % W;
        int64_t tmp = i / W;
        int64_t h = tmp % H;
        tmp = tmp / H;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        int64_t x_index = n * stride_x_n + c * stride_x_c + h * stride_x_h + w * stride_x_w;
        int64_t y_index = n * stride_y_n + c * stride_y_c + h * stride_y_h + w * stride_y_w;

        float val = x[x_index];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + (float)eps);
        val = (val - mean) * inv_std;
        if (weight != nullptr) {
            val *= weight[c];
        }
        if (bias != nullptr) {
            val += bias[c];
        }
        y[y_index] = val;
    }
}

// Kernel: softmax over channel dimension (dim=1)
// For each (n, h, w), compute softmax over c=0..C-1
// Input x and output y have same shape, but different strides possible
__global__ void softmax_channel_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w)
{
    // Grid-stride loop over spatial locations (n, h, w)
    int64_t total_spatial = N * H * W;
    for (int64_t flat_spatial = blockIdx.x; flat_spatial < total_spatial; flat_spatial += gridDim.x) {
        // decompose flat_spatial into n, h, w
        int64_t w = flat_spatial % W;
        int64_t tmp = flat_spatial / W;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        const float* x_base = x + n * stride_x_n + h * stride_x_h + w * stride_x_w;
        float* y_base = y + n * stride_y_n + h * stride_y_h + w * stride_y_w;

        // Step 1: find max over c
        float max_val = -INFINITY;
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            float val = x_base[c * stride_x_c];
            if (val > max_val) max_val = val;
        }

        // Block reduction to get global max in thread 0
        __shared__ float smem[1024];
        int tid = threadIdx.x;
        smem[tid] = max_val;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = smem[tid + s];
                if (other > smem[tid]) smem[tid] = other;
            }
            __syncthreads();
        }
        float block_max = smem[0];
        __syncthreads();

        // Step 2: compute sum of exp(x - max)
        float sum = 0.0f;
        for (int64_t c = tid; c < C; c += blockDim.x) {
            float val = x_base[c * stride_x_c];
            sum += expf(val - block_max);
        }

        // Reduce sum
        smem[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] += smem[tid + s];
            }
            __syncthreads();
        }
        float block_sum = smem[0];
        __syncthreads();

        // Step 3: compute softmax and write out
        // Avoid division by zero (if sum == 0, output 0)
        for (int64_t c = tid; c < C; c += blockDim.x) {
            float val = x_base[c * stride_x_c];
            float prob = (block_sum > 0.0f) ? expf(val - block_max) / block_sum : 0.0f;
            y_base[c * stride_y_c] = prob;
        }
        // Synchronize before next spatial iteration to avoid shared memory race
        __syncthreads();
    }
}

// Launcher for batch_norm2d_eval
extern "C" void batch_norm2d_eval_launcher(
    float* y,
    const float* x,
    const float* weight,  // nullable
    const float* bias,    // nullable
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream)
{
    int64_t total = N * C * H * W;
    if (total <= 0) return;

    const int block_size = 256;
    int64_t grid = (total + block_size - 1) / block_size;
    if (grid > 65535) grid = 65535;

    batch_norm2d_eval_kernel<<<(int)grid, block_size, 0, stream>>>(
        y, x, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w);
}

// Launcher for softmax_channel
extern "C" void softmax_channel_launcher(
    float* y,
    const float* x,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream)
{
    int64_t num_spatial = N * H * W;
    if (num_spatial <= 0 || C <= 0) return;

    // Choose block size: use at most 256 threads, but not more than C (power of 2 for reduction)
    int block_size = 1;
    int max_threads = (int)std::min<int64_t>(C, 256);
    while (block_size * 2 <= max_threads) block_size *= 2;
    if (block_size < 1) block_size = 1;

    // Grid size capped at 65535; kernel uses grid-stride loop over spatial locations
    int grid_size = (int)std::min<int64_t>(num_spatial, 65535);
    if (grid_size < 1) grid_size = 1;

    softmax_channel_kernel<<<grid_size, block_size, 0, stream>>>(
        y, x, N, C, H, W,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w);
}
