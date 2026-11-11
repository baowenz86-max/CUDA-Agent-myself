#include <cuda_runtime.h>
#include <math.h>

// SELU constants
#define SELU_SCALE 1.0507009873554804934193349852946
#define SELU_ALPHA 1.6732632423543772848170429916717

// InstanceNorm kernel: one block per (n,c), threads reduce over H*W
// x: input tensor with strides sN, sC, sH, sW (elements, not bytes)
// weight, bias: per-channel affine parameters length C
// output: same shape as input, with same strides
__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    float eps)
{
    int nc = blockIdx.x;
    int total = N * C;
    if (nc >= total) return;

    int n = nc / C;
    int c = nc - n * C;

    const float* x_base = x + (long long)n * sN + (long long)c * sC;
    float* out_base = out + (long long)n * sN + (long long)c * sC;
    int HW = H * W;

    // Compute mean and variance using shared memory reduction within block
    extern __shared__ float smem[];
    float* s_mean = smem;
    float* s_var = smem + blockDim.x;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int idx = threadIdx.x; idx < HW; idx += blockDim.x) {
        int h = idx / W;
        int w = idx - h * W;
        float val = x_base[(long long)h * sH + (long long)w * sW];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        s_mean[threadIdx.x >> 5] = local_sum;
        s_var[threadIdx.x >> 5] = local_sq_sum;
    }
    __syncthreads();

    // Now reduce partials from warps (only first few threads)
    int num_warps = blockDim.x / 32;
    if (threadIdx.x < num_warps) {
        local_sum = s_mean[threadIdx.x];
        local_sq_sum = s_var[threadIdx.x];
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset && threadIdx.x + offset < num_warps) {
                local_sum += s_mean[threadIdx.x + offset];
                local_sq_sum += s_var[threadIdx.x + offset];
                s_mean[threadIdx.x] = local_sum;
                s_var[threadIdx.x] = local_sq_sum;
            }
            __syncthreads();
        }
    }
    __syncthreads();

    float mean = s_mean[0] / HW;
    float var = s_var[0] / HW - mean * mean;
    float inv_std = rsqrtf(var + eps);
    float w = weight[c];
    float b = bias[c];

    // Normalize and apply affine
    for (int idx = threadIdx.x; idx < HW; idx += blockDim.x) {
        int h = idx / W;
        int w = idx - h * W;
        float val = x_base[(long long)h * sH + (long long)w * sW];
        out_base[(long long)h * sH + (long long)w * sW] = (val - mean) * inv_std * w + b;
    }
}

// SELU kernel: one element per thread, grid-stride
__global__ void selu_kernel(const float* __restrict__ x, float* __restrict__ y, long long numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float val = x[idx];
        y[idx] = val > 0.0f ? SELU_SCALE * val : SELU_SCALE * SELU_ALPHA * (expf(val) - 1.0f);
    }
}

// Launcher for instance_norm_forward
// Input tensors must be float32 on CUDA with strides provided. Handles contiguous or NHWC (strided) via strides.
extern "C" void instance_norm_forward_cuda_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    float eps,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int total = N * C;
    int threads = 256;
    int blocks = total; // one block per (n,c)
    size_t smem_size = 2 * threads * sizeof(float);

    instance_norm_kernel<<<blocks, threads, smem_size, stream>>>(
        x, weight, bias, out, N, C, H, W, sN, sC, sH, sW, eps);
}

// Launcher for selu_forward
extern "C" void selu_forward_cuda_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles larger
    selu_kernel<<<blocks, threads, 0, stream>>>(x, y, numel);
}
