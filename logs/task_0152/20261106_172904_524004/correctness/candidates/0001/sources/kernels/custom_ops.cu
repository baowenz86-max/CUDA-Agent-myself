#include <cuda_runtime.h>

// Helper: compute linear index from 4D indices and strides
__device__ __forceinline__ int64_t idx4(int64_t n, int64_t c, int64_t h, int64_t w,
                                        int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    return n * stride_n + c * stride_c + h * stride_h + w * stride_w;
}

// Elementwise hard sigmoid: y = clamp(x+3, 0, 6) / 6
__global__ void hardsigmoid_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t total_elements,
                                   int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                   int64_t N, int64_t C, int64_t H, int64_t W) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total_elements; i += stride) {
        // Decode linear index to n,c,h,w using row-major with given strides
        // Since total_elements = N*C*H*W, we can compute indices assuming contiguous storage order,
        // but using strides for actual memory offset.
        int64_t w = i % W;
        int64_t h = (i / W) % H;
        int64_t c = (i / (W * H)) % C;
        int64_t n = i / (W * H * C);
        int64_t idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
        float val = input[idx];
        float y = val + 3.0f;
        y = fminf(fmaxf(y, 0.0f), 6.0f);
        y *= (1.0f / 6.0f);
        output[idx] = y;
    }
}

// Kernel for computing mean and variance for each (n,c) pair.
// Each block handles one (n,c) pair, using shared memory reduction over H*W.
// Input is float, output is mean and var (as float2) stored in global arrays.
__global__ void instance_norm_stats_kernel(const float* __restrict__ input,
                                           float* __restrict__ mean,
                                           float* __restrict__ var,
                                           int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                           int64_t N, int64_t C, int64_t H, int64_t W) {
    // Assume blockIdx.x indexes n*C + c
    int64_t nc = blockIdx.x;
    if (nc >= N * C) return;
    int64_t n = nc / C;
    int64_t c = nc % C;
    int64_t base_offset = n * stride_n + c * stride_c;
    int64_t HW = H * W;

    extern __shared__ float sdata[]; // two arrays: [HW] for values, [HW] for squares? Actually we can compute sum and sum_sq in shared.
    // Better: use shared array of size 2*blockDim.x for reduction.
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;

    // Each thread computes partial sum and sum_sq over multiple elements
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < HW; i += blockDim.x) {
        // Compute h,w from i
        int64_t h = i / W;
        int64_t w = i % W;
        int64_t idx = base_offset + h * stride_h + w * stride_w;
        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Reduce within block using warp shuffle and shared memory
    // Standard reduction to sum all partial sums
    // We'll reduce local_sum and local_sum_sq together using shared memory
    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total_sum = s_sum[0];
        float total_sum_sq = s_sum_sq[0];
        float m = total_sum / (float)HW;
        float v = total_sum_sq / (float)HW - m * m;
        // Ensure non-negative variance due to floating error
        v = fmaxf(v, 0.0f);
        mean[nc] = m;
        var[nc] = v;
    }
}

// Kernel for normalization given mean and var
__global__ void instance_norm_normalize_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               const float* __restrict__ mean,
                                               const float* __restrict__ var,
                                               float eps,
                                               int64_t total_elements,
                                               int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                               int64_t N, int64_t C, int64_t H, int64_t W) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total_elements; i += stride) {
        int64_t w = i % W;
        int64_t h = (i / W) % H;
        int64_t c = (i / (W * H)) % C;
        int64_t n = i / (W * H * C);
        int64_t nc = n * C + c;
        int64_t idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
        float m = mean[nc];
        float v = var[nc];
        float inv_std = rsqrtf(v + eps);
        float val = input[idx];
        output[idx] = (val - m) * inv_std;
    }
}

// Launcher for hard sigmoid
extern "C" void custom_hardsigmoid_launcher(const float* input,
                                            float* output,
                                            int64_t total_elements,
                                            int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                            int64_t N, int64_t C, int64_t H, int64_t W,
                                            cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    hardsigmoid_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, total_elements, stride_n, stride_c, stride_h, stride_w, N, C, H, W);
}

// Launcher for instance norm (two stages)
extern "C" void custom_instance_norm2d_no_affine_launcher(const float* input,
                                                           float* output,
                                                           float eps,
                                                           int64_t N, int64_t C, int64_t H, int64_t W,
                                                           int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                                           cudaStream_t stream) {
    int64_t total_elements = N * C * H * W;
    // Allocate temporary memory for mean and var (NC elements each)
    float *d_mean, *d_var;
    cudaMalloc(&d_mean, sizeof(float) * N * C);
    cudaMalloc(&d_var, sizeof(float) * N * C);

    // Stage 1: compute stats, one block per (n,c)
    int block_size = 256;
    int grid_size = N * C;
    int shared_mem_size = 2 * block_size * sizeof(float); // for sum and sum_sq
    instance_norm_stats_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, d_mean, d_var, stride_n, stride_c, stride_h, stride_w, N, C, H, W);

    // Stage 2: normalize
    int64_t blocks_norm = (total_elements + block_size - 1) / block_size;
    instance_norm_normalize_kernel<<<blocks_norm, block_size, 0, stream>>>(
        input, output, d_mean, d_var, eps, total_elements,
        stride_n, stride_c, stride_h, stride_w, N, C, H, W);

    // Free temporary memory
    cudaFree(d_mean);
    cudaFree(d_var);
}
