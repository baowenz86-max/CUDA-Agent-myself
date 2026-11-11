#include <cuda_runtime.h>
#include <cstdint>

// Helper macro for error check after kernel launch
#define CUDA_CHECK_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            return; \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// Kernel: custom_gt_scalar
// Elementwise x > threshold, output bool
// Input: float x, output bool, scalar threshold
// -----------------------------------------------------------------------------
__global__ void gt_scalar_kernel(const float* __restrict__ x, bool* __restrict__ out,
                                 float threshold, int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = (x[i] > threshold);
    }
}

extern "C" void custom_gt_scalar_launcher(const float* x, bool* out, double threshold,
                                           int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    gt_scalar_kernel<<<blocks, threads, 0, stream>>>(x, out, (float)threshold, numel);
    CUDA_CHECK_LAUNCH();
}

// -----------------------------------------------------------------------------
// Kernel: custom_logical_and
// Elementwise a && b for bool tensors, output bool
// -----------------------------------------------------------------------------
__global__ void logical_and_kernel(const bool* __restrict__ a, const bool* __restrict__ b,
                                   bool* __restrict__ out, int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = a[i] && b[i];
    }
}

extern "C" void custom_logical_and_launcher(const bool* a, const bool* b, bool* out,
                                            int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    logical_and_kernel<<<blocks, threads, 0, stream>>>(a, b, out, numel);
    CUDA_CHECK_LAUNCH();
}

// -----------------------------------------------------------------------------
// Kernel: custom_where_mask
// out = x if mask else 0, mask shape [B,1,H,W], x shape [B,C,H,W]
// We assume mask data has same batch,spatial layout as x but with channel stride 0
// We pass strides for both mask and x, and output has same as x.
// dims: B, C, H, W (4D only, as per contract)
// mask_stride[0..3] and x_stride[0..3] are in elements
// -----------------------------------------------------------------------------
__global__ void where_mask_kernel(const float* __restrict__ x, const bool* __restrict__ mask,
                                  float* __restrict__ out,
                                  int B, int C, int H, int W,
                                  const int64_t* __restrict__ x_strides,
                                  const int64_t* __restrict__ mask_strides,
                                  const int64_t* __restrict__ out_strides) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * C * H * W;
    for (int64_t i = idx; i < total; i += stride) {
        // decompose i into b,c,h,w
        int b = (int)(i / ((int64_t)C * H * W));
        int rem = (int)(i % ((int64_t)C * H * W));
        int c = rem / (H * W);
        int rem2 = rem % (H * W);
        int h = rem2 / W;
        int w = rem2 % W;
        int64_t x_idx = b * x_strides[0] + c * x_strides[1] + h * x_strides[2] + w * x_strides[3];
        int64_t mask_idx = b * mask_strides[0] + 0 * mask_strides[1] + h * mask_strides[2] + w * mask_strides[3]; // channel stride ignored for mask
        int64_t out_idx = b * out_strides[0] + c * out_strides[1] + h * out_strides[2] + w * out_strides[3];
        out[out_idx] = mask[mask_idx] ? x[x_idx] : 0.0f;
    }
}

extern "C" void custom_where_mask_launcher(const float* x, const bool* mask, float* out,
                                           int B, int C, int H, int W,
                                           const int64_t* x_strides, const int64_t* mask_strides,
                                           const int64_t* out_strides, cudaStream_t stream) {
    int64_t total = (int64_t)B * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    where_mask_kernel<<<blocks, threads, 0, stream>>>(x, mask, out, B, C, H, W,
                                                       x_strides, mask_strides, out_strides);
    CUDA_CHECK_LAUNCH();
}

// -----------------------------------------------------------------------------
// Kernel: compute max and any_gt over dim=1 for each (b,h,w)
// Output: max_values [B,1,H,W] float, max_indices [B,1,H,W] int64,
// any_mask [B,1,H,W] bool
// We do both reductions in one kernel to improve efficiency, but also provide separate
// launchers for custom_max_dim and custom_any_gt.
// x has shape [B,C,H,W] with strides.
// We assume reduce dim = 1 always (as per contract).
// We output using keepdim=True (so shape [B,1,H,W]) with strides computed for
// keepdim layout: output strides may be arbitrary, passed by launcher.
// For simplicity, the launcher allocates output tensors contiguous, but we handle
// arbitrary output strides too (passed as out_val_strides, out_idx_strides, out_any_strides).
// Note: indices are int64_t.
// -----------------------------------------------------------------------------
__global__ void max_any_reduce_kernel(const float* __restrict__ x,
                                      float* __restrict__ max_values,
                                      int64_t* __restrict__ max_indices,
                                      bool* __restrict__ any_mask,
                                      int B, int C, int H, int W,
                                      const int64_t* __restrict__ x_strides,
                                      const int64_t* __restrict__ out_val_strides,
                                      const int64_t* __restrict__ out_idx_strides,
                                      const int64_t* __restrict__ out_any_strides,
                                      float any_threshold) {
    int64_t bhw_total = (int64_t)B * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t pos = idx; pos < bhw_total; pos += stride) {
        int b = (int)(pos / ((int64_t)H * W));
        int rem = (int)(pos % ((int64_t)H * W));
        int h = rem / W;
        int w = rem % W;

        float max_val = -3.402823466e+38f; // -FLT_MAX
        int64_t max_idx = 0;
        bool any_gt = false;

        for (int c = 0; c < C; ++c) {
            int64_t x_idx = b * x_strides[0] + c * x_strides[1] + h * x_strides[2] + w * x_strides[3];
            float val = x[x_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
            if (val > any_threshold) {
                any_gt = true;
            }
        }

        int64_t out_val_idx = b * out_val_strides[0] + 0 * out_val_strides[1] + h * out_val_strides[2] + w * out_val_strides[3];
        max_values[out_val_idx] = max_val;

        int64_t out_idx_idx = b * out_idx_strides[0] + 0 * out_idx_strides[1] + h * out_idx_strides[2] + w * out_idx_strides[3];
        max_indices[out_idx_idx] = max_idx;

        int64_t out_any_idx = b * out_any_strides[0] + 0 * out_any_strides[1] + h * out_any_strides[2] + w * out_any_strides[3];
        any_mask[out_any_idx] = any_gt;
    }
}

// Launcher for custom_max_dim (returns values and indices)
extern "C" void custom_max_dim_launcher(const float* x,
                                        float* max_values, int64_t* max_indices,
                                        int B, int C, int H, int W,
                                        const int64_t* x_strides,
                                        const int64_t* out_val_strides,
                                        const int64_t* out_idx_strides,
                                        cudaStream_t stream) {
    int64_t bhw_total = (int64_t)B * H * W;
    int threads = 256;
    int blocks = (int)((bhw_total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    // We ignore any_mask in this launcher by passing nullptr and any_threshold irrelevant
    max_any_reduce_kernel<<<blocks, threads, 0, stream>>>(x, max_values, max_indices, nullptr,
                                                          B, C, H, W, x_strides,
                                                          out_val_strides, out_idx_strides, nullptr, 0.0f);
    CUDA_CHECK_LAUNCH();
}

// Launcher for custom_any_gt (returns bool mask)
extern "C" void custom_any_gt_launcher(const float* x,
                                       bool* any_mask,
                                       int B, int C, int H, int W,
                                       const int64_t* x_strides,
                                       const int64_t* out_any_strides,
                                       double any_threshold,
                                       cudaStream_t stream) {
    int64_t bhw_total = (int64_t)B * H * W;
    int threads = 256;
    int blocks = (int)((bhw_total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_any_reduce_kernel<<<blocks, threads, 0, stream>>>(x, nullptr, nullptr, any_mask,
                                                          B, C, H, W, x_strides,
                                                          nullptr, nullptr, out_any_strides,
                                                          (float)any_threshold);
    CUDA_CHECK_LAUNCH();
}

// -----------------------------------------------------------------------------
// Kernel: custom_threshold_mask (fused)
// Computes max over C and any_gt over C per (b,h,w), then applies mask to all C channels.
// Equivalent to reference forward: max_mask = (max > max_threshold),
// any_mask = any(x > any_threshold), combined = max_mask && any_mask,
// out[b,c,h,w] = (combined[b,0,h,w] ? x[b,c,h,w] : 0)
// -----------------------------------------------------------------------------
__global__ void threshold_mask_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int B, int C, int H, int W,
                                      const int64_t* __restrict__ x_strides,
                                      const int64_t* __restrict__ out_strides,
                                      float max_threshold,
                                      float any_threshold) {
    int64_t bhw_total = (int64_t)B * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t pos = idx; pos < bhw_total; pos += stride) {
        int b = (int)(pos / ((int64_t)H * W));
        int rem = (int)(pos % ((int64_t)H * W));
        int h = rem / W;
        int w = rem % W;

        float max_val = -3.402823466e+38f;
        bool any_gt = false;
        for (int c = 0; c < C; ++c) {
            int64_t x_idx = b * x_strides[0] + c * x_strides[1] + h * x_strides[2] + w * x_strides[3];
            float val = x[x_idx];
            if (val > max_val) max_val = val;
            if (val > any_threshold) any_gt = true;
        }
        bool mask = (max_val > max_threshold) && any_gt;
        if (mask) {
            for (int c = 0; c < C; ++c) {
                int64_t x_idx = b * x_strides[0] + c * x_strides[1] + h * x_strides[2] + w * x_strides[3];
                int64_t out_idx = b * out_strides[0] + c * out_strides[1] + h * out_strides[2] + w * out_strides[3];
                out[out_idx] = x[x_idx];
            }
        } else {
            for (int c = 0; c < C; ++c) {
                int64_t out_idx = b * out_strides[0] + c * out_strides[1] + h * out_strides[2] + w * out_strides[3];
                out[out_idx] = 0.0f;
            }
        }
    }
}

extern "C" void custom_threshold_mask_launcher(const float* x, float* out,
                                                int B, int C, int H, int W,
                                                const int64_t* x_strides,
                                                const int64_t* out_strides,
                                                double max_threshold, double any_threshold,
                                                cudaStream_t stream) {
    int64_t bhw_total = (int64_t)B * H * W;
    int threads = 256;
    int blocks = (int)((bhw_total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    threshold_mask_kernel<<<blocks, threads, 0, stream>>>(x, out, B, C, H, W,
                                                          x_strides, out_strides,
                                                          (float)max_threshold, (float)any_threshold);
    CUDA_CHECK_LAUNCH();
}
