// CUDA kernel for instance normalization forward
// Simple two-pass: compute sum and sum_sq, then normalize
#include <cuda_runtime.h>

__global__ void instance_norm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float eps
) {
    int instance_idx = blockIdx.x;
    int n = instance_idx / C;
    int c = instance_idx % C;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* x_base = x + n * in_stride_n + c * in_stride_c;
    float* y_base = y + n * out_stride_n + c * out_stride_c;

    int numel = H * W;

    // Partial sums per thread
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int idx = tid; idx < numel; idx += block_size) {
        int h = idx / W;
        int w = idx - h * W;
        float val = x_base[h * in_stride_h + w * in_stride_w];
        sum += val;
        sum_sq += val * val;
    }

    // Shared memory reduction for sum and sum_sq
    __shared__ float sh_sum[256];
    __shared__ float sh_sum_sq[256];
    sh_sum[tid] = sum;
    sh_sum_sq[tid] = sum_sq;
    __syncthreads();

    // Reduce
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_sum_sq[tid] += sh_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    float total_sum = sh_sum[0];
    float total_sum_sq = sh_sum_sq[0];
    float mean = total_sum / numel;
    float var = total_sum_sq / numel - mean * mean;
    if (var < 0) var = 0;
    float inv_std = rsqrtf(var + eps);

    float w = weight[c];
    float b = bias[c];

    // Apply normalization
    for (int idx = tid; idx < numel; idx += block_size) {
        int h = idx / W;
        int w = idx - h * W;
        float val = x_base[h * in_stride_h + w * in_stride_w];
        float normalized = (val - mean) * inv_std;
        y_base[h * out_stride_h + w * out_stride_w] = normalized * w + b;
    }
}

// Host launcher
// Same signature as before
extern "C" void instance_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float eps,
    cudaStream_t stream
) {
    int num_instances = N * C;
    int block_size = 256;
    dim3 grid(num_instances);
    instance_norm_forward_kernel<<<grid, block_size, 0, stream>>>(
        x, weight, bias, y, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        eps
    );
}

// abs_forward kernel remains unchanged
__global__ void abs_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long numel
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        y[i] = fabsf(x[i]);
    }
}

extern "C" void abs_forward_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (int)min((long long)65535, (numel + block_size - 1) / block_size);
    if (grid_size <= 0) grid_size = 1;
    abs_forward_kernel<<<grid_size, block_size, 0, stream>>>(x, y, numel);
}
