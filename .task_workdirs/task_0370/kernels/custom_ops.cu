#include <cuda_runtime.h>
#include <cstdio>

// GroupNorm forward kernel: computes mean and biased variance per (n,c) group.
// Since num_groups == num_channels, each group is one channel.
// gridDim.x covers N*C pairs, blockDim.x covers H*W elements.
__global__ void group_norm_mean_var_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ var,
    int H,
    int W,
    int C,
    int HW,
    int N)
{
    const int nc = blockIdx.x;
    if (nc >= N * C) return;
    const int n = nc / C;
    const int c = nc % C;
    const float* x_nc = x + (size_t)n * C * HW + (size_t)c * HW;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_nc[i];
        sum += val;
        sumsq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }

    __shared__ float s_sum[32];
    __shared__ float s_sumsq[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        s_sum[warp] = sum;
        s_sumsq[warp] = sumsq;
    }
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0.0f;
        sumsq = (lane < (blockDim.x >> 5)) ? s_sumsq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
        }
        if (lane == 0) {
            mean[nc] = sum / HW;
            var[nc] = fmaxf(sumsq / HW - mean[nc] * mean[nc], 0.0f);
        }
    }
}

// GroupNorm forward kernel: normalizes and applies affine.
__global__ void group_norm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int H,
    int W,
    int C,
    int HW,
    int N,
    double eps)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * HW) return;
    const int nc = idx / HW;
    const int hw = idx - nc * HW;
    const int c = nc % C;
    float mu = mean[nc];
    float sigma = sqrtf(var[nc] + (float)eps);
    float val = x[idx];
    y[idx] = (val - mu) / sigma * weight[c] + bias[c];
}

// Launcher for group_norm_forward. Parameters are raw pointers on CUDA device.
extern "C" void group_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N,
    int C,
    int H,
    int W,
    double eps,
    cudaStream_t stream)
{
    const int HW = H * W;
    const int NC = N * C;

    // Allocate temporary buffers for mean and var
    float *d_mean, *d_var;
    cudaError_t err;
    err = cudaMalloc(&d_mean, NC * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc mean failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_var, NC * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc var failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_mean);
        return;
    }

    // Launch mean/var kernel: one block per (n,c) pair. Block size 256.
    const int block = 256;
    const int grid_mean = NC;
    group_norm_mean_var_kernel<<<grid_mean, block, 0, stream>>>(
        x, d_mean, d_var, H, W, C, HW, N);

    // Launch normalization kernel
    const int total = N * C * HW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    group_norm_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, d_mean, d_var, weight, bias, y, H, W, C, HW, N, eps);

    // Free temporary buffers asynchronously? We can free immediately since kernels are queued and we don't need d_mean/d_var after kernel uses them,
    // but to be safe, we can use cudaFree after synchronizing or use cudaStreamSynchronize.
    // For simplicity, synchronize and then free.
    cudaStreamSynchronize(stream);
    cudaFree(d_mean);
    cudaFree(d_var);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after group_norm_forward: %s\n", cudaGetErrorString(err));
    }
}
