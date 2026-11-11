// CUDA kernels for custom_mean and custom_layer_norm
#include <cuda_runtime.h>
#include <cstdint>

// --------------------------- custom_mean ---------------------------

// Kernel: compute mean over dimension 1 (features) with keepdim.
// Input x: shape [B, F, H, W], arbitrary strides.
// Output y: shape [B, 1, H, W], arbitrary strides.
// We use one thread per output element (b, h, w) to sum over features.
// Grid-stride loop over total output elements = B*H*W.
template <typename T>
__global__ void mean_dim1_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t B,
    int64_t F,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_f,
    int64_t sx_h,
    int64_t sx_w,
    int64_t sy_b,
    int64_t sy_h,
    int64_t sy_w)
{
    int64_t total = B * H * W;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x)
    {
        int64_t b = idx / (H * W);
        int64_t rem = idx % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        const T* base = x + b * sx_b + h * sx_h + w * sx_w;
        T sum = 0;
        for (int64_t f = 0; f < F; ++f) {
            sum += base[f * sx_f];
        }
        T mean = sum / (T)F;
        y[b * sy_b + h * sy_h + w * sy_w] = mean;
    }
}

// Launcher for custom_mean
// Input: x pointer, B,F,H,W, strides sx_*, output strides sy_*, stream
extern "C" void custom_mean_launcher(
    const float* x,
    float* y,
    int64_t B,
    int64_t F,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_f,
    int64_t sx_h,
    int64_t sx_w,
    int64_t sy_b,
    int64_t sy_h,
    int64_t sy_w,
    cudaStream_t stream)
{
    if (B <= 0 || F <= 0 || H <= 0 || W <= 0) return;
    int64_t total = B * H * W;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to 2^31-1; use grid-stride loop to handle larger.
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    mean_dim1_kernel<float><<<(unsigned int)grid_size, block_size, 0, stream>>>(
        x, y, B, F, H, W, sx_b, sx_f, sx_h, sx_w, sy_b, sy_h, sy_w);
}

// ------------------------ custom_layer_norm ------------------------
// Input x: shape [B, 1, H, W], arbitrary strides.
// normalized_shape values are not passed to launcher; we use H,W and assume normalized_shape = (1,H,W).
// Output y: same shape, same strides as input (or arbitrary provided strides).
// For each batch b, compute mean and variance over H*W elements, then normalize.
// Two-pass approach: first kernel computes per-batch statistics, second kernel normalizes.

// Kernel to compute sum and sum of squares per batch over H*W spatial elements.
// One block per batch item, block threads reduce spatial elements.
template <typename T>
__global__ void ln_stats_kernel(
    const T* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ inv_std, // store 1/std after var, to avoid recomputation
    int64_t B,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_h,
    int64_t sx_w,
    float eps)
{
    extern __shared__ float sdata[]; // [blockDim.x] for reduction, actually two arrays? use one per thread temp
    // We can use shared memory for reduction of sum and sumsq.
    // But simpler: each thread accumulates its partial for its assigned elements,
    // then reduce across block using warp shuffles or shared memory.
    int64_t b = blockIdx.x;
    if (b >= B) return;

    const T* base = x + b * sx_b;
    int64_t spatial = H * W;
    float thread_sum = 0.f;
    float thread_sumsq = 0.f;

    // Grid-stride within spatial with all threads in block
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    for (int64_t i = tid; i < spatial; i += nthreads) {
        int64_t h = i / W;
        int64_t w = i % W;
        float val = (float)base[h * sx_h + w * sx_w];
        thread_sum += val;
        thread_sumsq += val * val;
    }

    // Reduce thread_sum and thread_sumsq within block
    // Use shared memory arrays separate for sum and sumsq
    __shared__ float s_sum[1024];   // blockDim.x up to 1024
    __shared__ float s_sumsq[1024];
    s_sum[tid] = thread_sum;
    s_sumsq[tid] = thread_sumsq;
    __syncthreads();

    // Reduction in shared memory
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sumsq[tid] += s_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = s_sum[0];
        float sumsq = s_sumsq[0];
        float mu = sum / (float)spatial;
        float var = sumsq / (float)spatial - mu * mu;
        // Clamp variance to non-negative for numerical stability
        if (var < 0.f) var = 0.f;
        float std = sqrtf(var + eps);
        mean[b] = mu;
        inv_std[b] = 1.f / std;
    }
}

// Kernel to normalize each element using per-batch statistics.
// One thread per element (b, h, w), grid-stride loop.
template <typename T>
__global__ void ln_apply_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    int64_t B,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_h,
    int64_t sx_w,
    int64_t sy_b,
    int64_t sy_h,
    int64_t sy_w)
{
    int64_t total = B * H * W;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x)
    {
        int64_t b = idx / (H * W);
        int64_t rem = idx % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        float val = (float)x[b * sx_b + h * sx_h + w * sx_w];
        float norm = (val - mean[b]) * inv_std[b];
        y[b * sy_b + h * sy_h + w * sy_w] = (T)norm;
    }
}

extern "C" void custom_layer_norm_launcher(
    const float* x,
    float* y,
    int64_t B,
    int64_t H,
    int64_t W,
    int64_t sx_b,
    int64_t sx_h,
    int64_t sx_w,
    int64_t sy_b,
    int64_t sy_h,
    int64_t sy_w,
    float eps,
    cudaStream_t stream)
{
    if (B <= 0 || H <= 0 || W <= 0) return;

    // Allocate temporary per-batch statistics on device
    // Use cudaMallocAsync? Use cudaMalloc for simplicity (binding will pass stream,
    // but we can just use cudaMalloc from host, asynchronous alloc may stall).
    // To avoid host sync, we can use cudaMalloc and rely on stream ordering.
    float* d_mean = nullptr;
    float* d_inv_std = nullptr;
    cudaMalloc(&d_mean, B * sizeof(float));
    cudaMalloc(&d_inv_std, B * sizeof(float));

    // First kernel: compute statistics
    int block_size = 256;
    if (block_size > 1024) block_size = 1024;
    // Use grid dimension = B (one block per batch)
    unsigned int grid_stats = (unsigned int)std::min<int64_t>(B, 2147483647);
    size_t shared_mem = block_size * sizeof(float); // only one shared array for reduction? We have two arrays, need 2 * block_size * sizeof(float)
    // Actually s_sum and s_sumsq are separate arrays each block_size, so total 2*block_size*4 bytes.
    shared_mem = 2 * block_size * sizeof(float);
    ln_stats_kernel<float><<<grid_stats, block_size, shared_mem, stream>>>(
        x, d_mean, d_inv_std, B, H, W, sx_b, sx_h, sx_w, eps);

    // Second kernel: normalize
    int64_t total = B * H * W;
    int block_apply = 256;
    int64_t grid_apply = (total + block_apply - 1) / block_apply;
    if (grid_apply > 2147483647LL) grid_apply = 2147483647LL;
    ln_apply_kernel<float><<<(unsigned int)grid_apply, block_apply, 0, stream>>>(
        x, y, d_mean, d_inv_std, B, H, W, sx_b, sx_h, sx_w, sy_b, sy_h, sy_w);

    // Free temporary buffers after kernel completes on stream
    cudaFree(d_mean);
    cudaFree(d_inv_std);
}

// Note: The binding.cpp must call these launchers and handle tensor allocation,
// strides, normalized_shape parsing, and error checks. The launchers assume valid pointers.
