#include <cuda_runtime.h>
#include <math.h>

// GELU forward kernel (exact PyTorch F.gelu with approximate='none')
// Formula: 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)))
__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    long long numel) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = x[i];
        y[i] = 0.5f * val * (1.0f + erff(val * 0.70710678118654752440f));
    }
}

// InstanceNorm2d forward kernel for affine=False, track_running_stats=False.
// Each block processes one (N, C) plane, computing spatial mean/var and then normalizing.
// Input layout: contiguous NCHW, so plane size = H * W, plane stride = H * W.
// Grid dimension: N * C blocks.
__global__ void instance_norm2d_no_affine_kernel(const float* __restrict__ x,
                                                 float* __restrict__ y,
                                                 int plane_size,
                                                 float eps) {
    extern __shared__ float shared_mem[];
    float* sum_shared = shared_mem;          // length blockDim.x
    float* sumsq_shared = &shared_mem[blockDim.x]; // length blockDim.x

    int tid = threadIdx.x;
    int plane = blockIdx.x;
    const float* x_plane = x + (long long)plane * plane_size;
    float* y_plane = y + (long long)plane * plane_size;

    // First pass: compute sum and sum of squares using strided loop.
    float sum = 0.0f;
    float sumsq = 0.0f;
    int idx = tid;
    while (idx < plane_size) {
        float val = x_plane[idx];
        sum += val;
        sumsq += val * val;
        idx += blockDim.x;
    }
    sum_shared[tid] = sum;
    sumsq_shared[tid] = sumsq;
    __syncthreads();

    // Reduce within block (blockDim.x is power of 2, <= 1024)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
            sumsq_shared[tid] += sumsq_shared[tid + s];
        }
        __syncthreads();
    }

    float total_sum = sum_shared[0];
    float total_sumsq = sumsq_shared[0];
    float mean = total_sum / plane_size;
    float var = total_sumsq / plane_size - mean * mean;
    // Clamp var to non-negative (numerical safety)
    var = fmaxf(var, 0.0f);
    float inv_std = rsqrtf(var + eps);

    // Second pass: normalize and write output.
    idx = tid;
    while (idx < plane_size) {
        y_plane[idx] = (x_plane[idx] - mean) * inv_std;
        idx += blockDim.x;
    }
}

// C-interface launchers (no PyTorch dependencies).
// All pointers must be CUDA device pointers; stream is provided by binding.
extern "C" void gelu_forward_launcher(const float* x,
                                      float* y,
                                      long long numel,
                                      cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    int blocks = (int)(blocks_ll > 0 ? blocks_ll : 1);
    gelu_forward_kernel<<<blocks, threads, 0, stream>>>(x, y, numel);
}

extern "C" void instance_norm2d_no_affine_launcher(const float* x,
                                                    float* y,
                                                    int num_planes,
                                                    int plane_size,
                                                    float eps,
                                                    cudaStream_t stream) {
    int threads = 256;
    // Ensure threads <= plane_size for proper reduction, but handle plane_size=1 etc.
    if (plane_size < threads) {
        threads = 1;
        while (threads * 2 <= plane_size) threads *= 2;
        if (threads > 1024) threads = 1024; // max block size
    }
    // Shared memory: 2 * threads * sizeof(float)
    size_t shared_mem_size = 2 * threads * sizeof(float);
    instance_norm2d_no_affine_kernel<<<num_planes, threads, shared_mem_size, stream>>>(
        x, y, plane_size, eps);
}
