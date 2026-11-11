// kernels/custom_ops.cu
// CUDA kernels for custom_abs and custom_layer_norm
// No PyTorch dependencies; launchers use raw pointers and cudaStream_t

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// custom_abs kernel: y[i] = |x[i]|
// Input/output tensors may have arbitrary strides; we linearize over numel.
// Grid-stride loop for large sizes.
// -----------------------------------------------------------------------------
__global__ void abs_kernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        y[i] = fabsf(x[i]);
    }
}

// -----------------------------------------------------------------------------
// custom_layer_norm kernel
// Normalizes over last 3 dims (C,H,W), which are contiguous in memory for
// standard NCHW layout, but we must respect strides just in case.
// For each sample (batch index b), we compute mean and variance over the
// normalized region of size norm_size = C*H*W.
// Then apply (x - mean) / sqrt(var + eps) * weight + bias elementwise.
// We use the standard two-pass algorithm with double precision accumulation.
// Block reduction is used; each block handles one sample.
// gridDim.x = batch_size, blockDim.x = reduction block size.
// -----------------------------------------------------------------------------
__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    float eps,
    int64_t batch_size,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t norm_size,          // C*H*W
    int64_t spatial_stride_b,   // stride of batch dimension
    int64_t spatial_stride_c,   // stride of channel dimension
    int64_t spatial_stride_h,   // stride of height
    int64_t spatial_stride_w,   // stride of width
    int64_t weight_stride_c,    // strides for weight/bias (same shape)
    int64_t weight_stride_h,
    int64_t weight_stride_w,
    bool has_affine) {

    extern __shared__ double shared_double[]; // [blockDim.x] for sum, [blockDim.x] for sum_sq
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Base pointer for this sample in x and y
    const float* x_b = x + b * spatial_stride_b;
    float* y_b = y + b * norm_size;  // output is contiguous

    double* s_sum = shared_double;
    double* s_sum_sq = shared_double + n_threads;

    // -- Phase 1: compute local sum and sum of squares --
    double local_sum = 0.0;
    double local_sum_sq = 0.0;

    // Assign contiguous chunks for deterministic coverage.
    // The last thread may get a few extra elements.
    int64_t base = (norm_size / n_threads) * tid;
    int64_t end = (tid == n_threads - 1) ? norm_size : base + (norm_size / n_threads);

    for (int64_t idx = base; idx < end; ++idx) {
        // Compute multi-dimensional indices for this sample
        int64_t rem = idx;
        int w = rem % W;
        rem /= W;
        int h = rem % H;
        rem /= H;
        int c = rem;  // c < C guaranteed by idx < norm_size

        float val = x_b[c * spatial_stride_c + h * spatial_stride_h + w * spatial_stride_w];
        local_sum += (double)val;
        local_sum_sq += (double)val * (double)val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // -- Phase 2: block reduction --
    for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    double total_sum = s_sum[0];
    double total_sum_sq = s_sum_sq[0];
    double mean = total_sum / (double)norm_size;
    double var = total_sum_sq / (double)norm_size - mean * mean;
    // Clamp variance to non-negative due to floating point error
    if (var < 0.0) var = 0.0;
    double inv_std = rsqrt((float)(var + (double)eps));

    // -- Phase 3: normalize and apply affine --
    for (int64_t idx = base; idx < end; ++idx) {
        int64_t rem = idx;
        int w = rem % W;
        rem /= W;
        int h = rem % H;
        rem /= H;
        int c = rem;

        float x_val = x_b[c * spatial_stride_c + h * spatial_stride_h + w * spatial_stride_w];
        float normalized = (float)(((double)x_val - mean) * inv_std);

        if (has_affine) {
            float w_val = weight[c * weight_stride_c + h * weight_stride_h + w * weight_stride_w];
            float b_val = bias[c * weight_stride_c + h * weight_stride_h + w * weight_stride_w];
            normalized = normalized * w_val + b_val;
        }
        y_b[idx] = normalized; // output is contiguous
    }
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C" to be called from binding.cpp)
// -----------------------------------------------------------------------------
extern "C" void custom_abs_launcher(
    const float* x,
    float* y,
    int64_t numel,
    cudaStream_t stream) {

    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    abs_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, numel);
}

extern "C" void custom_layer_norm_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    double eps,
    int64_t batch_size,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t norm_size,
    int64_t spatial_stride_b,
    int64_t spatial_stride_c,
    int64_t spatial_stride_h,
    int64_t spatial_stride_w,
    int64_t weight_stride_c,
    int64_t weight_stride_h,
    int64_t weight_stride_w,
    bool has_affine,
    cudaStream_t stream) {

    const int block_size = 256;
    if (batch_size <= 0 || norm_size <= 0) return;

    // Shared memory for double sums: 2 * block_size * sizeof(double)
    size_t shared_mem_size = 2 * block_size * sizeof(double);

    layer_norm_kernel<<<(int)batch_size, block_size, shared_mem_size, stream>>>(
        x, weight, bias, y, (float)eps, batch_size, C, H, W, norm_size,
        spatial_stride_b, spatial_stride_c, spatial_stride_h, spatial_stride_w,
        weight_stride_c, weight_stride_h, weight_stride_w, has_affine);
}
