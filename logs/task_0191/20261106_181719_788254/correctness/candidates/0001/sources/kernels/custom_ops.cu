#include <cuda_runtime.h>

// CUDA kernel for instance normalization forward
// Input: x (N,C,H,W) with arbitrary strides, weight[C], bias[C], eps
// Output: y (N,C,H,W) contiguous, same shape and strides as input (but output is allocated contiguous by binding)
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
    // Each block handles one (n,c) instance
    int instance_idx = blockIdx.x;
    int n = instance_idx / C;
    int c = instance_idx % C;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Base pointers for this instance
    const float* x_base = x + n * in_stride_n + c * in_stride_c;
    float* y_base = y + n * out_stride_n + c * out_stride_c;

    // Welford's online algorithm for mean and variance
    // We'll compute sum and sum_of_squares using a reduction within the block
    // First pass: each thread computes partial sums over its assigned elements
    int numel = H * W;
    float mean = 0.0f;
    float M2 = 0.0f;
    int count = 0;

    // Grid-stride loop over spatial elements
    for (int idx = tid; idx < numel; idx += block_size) {
        int h = idx / W;
        int w = idx - h * W;
        float val = x_base[h * in_stride_h + w * in_stride_w];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    // Block reduction of Welford accumulators (count, mean, M2)
    __shared__ float sh_mean[1024];
    __shared__ float sh_M2[1024];
    __shared__ int sh_count[1024];

    if (tid < block_size) {
        sh_mean[tid] = mean;
        sh_M2[tid] = M2;
        sh_count[tid] = count;
    }
    __syncthreads();

    // Reduce in shared memory using tree reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other = tid + stride;
            int count1 = sh_count[tid];
            int count2 = sh_count[other];
            float mean1 = sh_mean[tid];
            float mean2 = sh_mean[other];
            float M2_1 = sh_M2[tid];
            float M2_2 = sh_M2[other];
            int combined_count = count1 + count2;
            if (combined_count > 0) {
                float delta = mean2 - mean1;
                float combined_mean = (count1 * mean1 + count2 * mean2) / combined_count;
                float combined_M2 = M2_1 + M2_2 + delta * delta * (count1 * count2) / combined_count;
                sh_mean[tid] = combined_mean;
                sh_M2[tid] = combined_M2;
                sh_count[tid] = combined_count;
            } else {
                sh_mean[tid] = 0.0f;
                sh_M2[tid] = 0.0f;
                sh_count[tid] = 0;
            }
        }
        __syncthreads();
    }

    float total_mean = sh_mean[0];
    float total_variance = (sh_count[0] > 0) ? (sh_M2[0] / sh_count[0]) : 0.0f;
    float inv_std = rsqrtf(total_variance + eps);
    float w = weight[c];
    float b = bias[c];

    // Second pass: normalize and apply affine
    for (int idx = tid; idx < numel; idx += block_size) {
        int h = idx / W;
        int w = idx - h * W;
        float val = x_base[h * in_stride_h + w * in_stride_w];
        float normalized = (val - total_mean) * inv_std;
        y_base[h * out_stride_h + w * out_stride_w] = normalized * w + b;
    }
}

// Host launcher for instance_norm_forward
// This function is declared extern "C" so it can be called from C++ binding.
// Parameters:
//   x, weight, bias: input pointers (float)
//   y: output pointer (float, contiguous, same shape as x)
//   N, C, H, W: dimensions
//   strides for input and output (in elements)
//   eps: float
//   stream: CUDA stream
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
    // Use enough blocks to cover all instances, but each block handles one instance.
    // Grid can be large, but limited by max grid dimensions; N*C is typically small enough.
    dim3 grid(num_instances);
    instance_norm_forward_kernel<<<grid, block_size, 0, stream>>>(
        x, weight, bias, y, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        eps
    );
}

// CUDA kernel for abs_forward
// Applies fabsf to each element.
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

// Host launcher for abs_forward
extern "C" void abs_forward_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream
) {
    // Use a grid-stride loop to handle large tensors.
    int block_size = 256;
    int grid_size = (int)min((long long)65535, (numel + block_size - 1) / block_size);
    if (grid_size <= 0) grid_size = 1;
    abs_forward_kernel<<<grid_size, block_size, 0, stream>>>(x, y, numel);
}
