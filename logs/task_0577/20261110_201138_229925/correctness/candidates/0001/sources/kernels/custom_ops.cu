// kernels/custom_ops.cu
// CUDA kernels for custom_abs and custom_layer_norm
// No PyTorch dependencies; launchers use raw pointers and cudaStream_t

#include <cuda_runtime.h>
#include <cstdint>

// Helper for float4 vectorized loads (optional)
// We'll keep it simple, rely on coalescing.

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
// We use Welford's online algorithm for stable variance.
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
    int64_t norm_size,          // number of elements per sample to normalize (C*H*W)
    int64_t spatial_stride_b,   // stride of batch dimension (usually norm_size, but may differ)
    int64_t spatial_stride_c,   // stride of channel dimension
    int64_t spatial_stride_h,   // stride of height
    int64_t spatial_stride_w,   // stride of width
    int64_t weight_stride_c,    // strides for weight/bias (same shape)
    int64_t weight_stride_h,
    int64_t weight_stride_w,
    bool has_affine) {

    extern __shared__ float shared[]; // [blockDim.x] for reduction
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    int blockDim = blockDim.x;

    // Base pointer for this sample in x and y
    const float* x_b = x + b * spatial_stride_b;
    float* y_b = y + b * norm_size;  // output is contiguous, so batch stride = norm_size

    // -- Phase 1: compute mean and variance using Welford --
    float mean = 0.0f;
    float m2 = 0.0f;
    int64_t count = 0;

    // Grid-stride loop over norm_size elements (within the sample)
    for (int64_t idx = tid; idx < norm_size; idx += blockDim) {
        // Compute multi-dimensional indices for this sample
        int64_t rem = idx;
        int w = rem % spatial_stride_w;
        rem /= spatial_stride_w;
        int h = rem % spatial_stride_h;
        rem /= spatial_stride_h;
        int c = rem;  // remaining is channel index (assuming norm_size = C*H*W)

        float val = x_b[c * spatial_stride_c + h * spatial_stride_h + w];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }

    // -- Phase 2: block reduction of mean and m2 --
    // Use shared memory to reduce count, mean, m2 across threads
    float* s_count = shared;
    float* s_mean = shared + blockDim;
    float* s_m2 = shared + 2 * blockDim;

    s_count[tid] = (float)count;
    s_mean[tid] = mean;
    s_m2[tid] = m2;

    __syncthreads();

    // Reduce count first (could be combined but simpler separate)
    for (int stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_count = s_count[tid + stride];
            float other_mean = s_mean[tid + stride];
            float other_m2 = s_m2[tid + stride];
            float total_count = s_count[tid] + other_count;
            if (total_count > 0) {
                float delta = other_mean - s_mean[tid];
                float new_mean = s_mean[tid] + delta * other_count / total_count;
                float new_m2 = s_m2[tid] + other_m2 + delta * delta * s_count[tid] * other_count / total_count;
                s_mean[tid] = new_mean;
                s_m2[tid] = new_m2;
                s_count[tid] = total_count;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes final mean, variance
    float final_mean = s_mean[0];
    float final_var = s_m2[0] / (float)norm_size; // population variance
    float inv_std = rsqrtf(final_var + eps);

    // Store final mean and inv_std in shared memory for all threads
    s_mean[0] = final_mean;
    s_m2[0] = inv_std;
    __syncthreads();

    // -- Phase 3: normalize and apply affine --
    // Need to loop over norm_size elements again, same mapping
    for (int64_t idx = tid; idx < norm_size; idx += blockDim) {
        int64_t rem = idx;
        int w = rem % spatial_stride_w;
        rem /= spatial_stride_w;
        int h = rem % spatial_stride_h;
        rem /= spatial_stride_h;
        int c = rem;

        float x_val = x_b[c * spatial_stride_c + h * spatial_stride_h + w];
        float normalized = (x_val - final_mean) * inv_std;

        if (has_affine) {
            // weight and bias are contiguous with same shape as normalized dims
            float w_val = weight[c * weight_stride_c + h * weight_stride_h + w];
            float b_val = bias[c * weight_stride_c + h * weight_stride_h + w];
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
    if (grid_size > 65535) grid_size = 65535; // grid-stride loop handles rest
    abs_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, numel);
}

extern "C" void custom_layer_norm_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    double eps,
    int64_t batch_size,
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

    const int block_size = 256; // must be power of two for reduction
    if (batch_size <= 0 || norm_size <= 0) return;

    // Each block processes one sample.
    // Shared memory: 3 floats per thread for reduction arrays.
    size_t shared_mem_size = 3 * block_size * sizeof(float);

    layer_norm_kernel<<<(int)batch_size, block_size, shared_mem_size, stream>>>(
        x, weight, bias, y, (float)eps, batch_size, norm_size,
        spatial_stride_b, spatial_stride_c, spatial_stride_h, spatial_stride_w,
        weight_stride_c, weight_stride_h, weight_stride_w, has_affine);
}
