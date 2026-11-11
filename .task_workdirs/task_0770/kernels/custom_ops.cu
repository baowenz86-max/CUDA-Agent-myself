#include <cuda_runtime.h>
#include <cstdint>

#define CUDA_CHECK_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            return; \
        } \
    } while(0)

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

__global__ void where_mask_kernel(const float* __restrict__ x, const bool* __restrict__ mask,
                                  float* __restrict__ out,
                                  int B, int C, int H, int W,
                                  int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                  int64_t mask_s0, int64_t mask_s1, int64_t mask_s2, int64_t mask_s3,
                                  int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total = (int64_t)B * C * H * W;
    for (int64_t i = idx; i < total; i += stride) {
        int b = (int)(i / ((int64_t)C * H * W));
        int rem = (int)(i % ((int64_t)C * H * W));
        int c = rem / (H * W);
        int rem2 = rem % (H * W);
        int h = rem2 / W;
        int w = rem2 % W;
        int64_t x_idx = b * x_s0 + c * x_s1 + h * x_s2 + w * x_s3;
        int64_t mask_idx = b * mask_s0 + 0 * mask_s1 + h * mask_s2 + w * mask_s3;
        int64_t out_idx = b * out_s0 + c * out_s1 + h * out_s2 + w * out_s3;
        out[out_idx] = mask[mask_idx] ? x[x_idx] : 0.0f;
    }
}

extern "C" void custom_where_mask_launcher(const float* x, const bool* mask, float* out,
                                           int B, int C, int H, int W,
                                           int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                           int64_t mask_s0, int64_t mask_s1, int64_t mask_s2, int64_t mask_s3,
                                           int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3,
                                           cudaStream_t stream) {
    int64_t total = (int64_t)B * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    where_mask_kernel<<<blocks, threads, 0, stream>>>(x, mask, out, B, C, H, W,
                                                      x_s0, x_s1, x_s2, x_s3,
                                                      mask_s0, mask_s1, mask_s2, mask_s3,
                                                      out_s0, out_s1, out_s2, out_s3);
    CUDA_CHECK_LAUNCH();
}

__global__ void max_any_reduce_kernel(const float* __restrict__ x,
                                      float* __restrict__ max_values,
                                      int64_t* __restrict__ max_indices,
                                      bool* __restrict__ any_mask,
                                      int B, int C, int H, int W,
                                      int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                      int64_t out_val_s0, int64_t out_val_s1, int64_t out_val_s2, int64_t out_val_s3,
                                      int64_t out_idx_s0, int64_t out_idx_s1, int64_t out_idx_s2, int64_t out_idx_s3,
                                      int64_t out_any_s0, int64_t out_any_s1, int64_t out_any_s2, int64_t out_any_s3,
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
        int64_t max_idx = 0;
        bool any_gt = false;

        for (int c = 0; c < C; ++c) {
            int64_t x_idx = b * x_s0 + c * x_s1 + h * x_s2 + w * x_s3;
            float val = x[x_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
            if (val > any_threshold) {
                any_gt = true;
            }
        }

        if (max_values != nullptr) {
            int64_t out_val_idx = b * out_val_s0 + 0 * out_val_s1 + h * out_val_s2 + w * out_val_s3;
            max_values[out_val_idx] = max_val;
        }
        if (max_indices != nullptr) {
            int64_t out_idx_idx = b * out_idx_s0 + 0 * out_idx_s1 + h * out_idx_s2 + w * out_idx_s3;
            max_indices[out_idx_idx] = max_idx;
        }
        if (any_mask != nullptr) {
            int64_t out_any_idx = b * out_any_s0 + 0 * out_any_s1 + h * out_any_s2 + w * out_any_s3;
            any_mask[out_any_idx] = any_gt;
        }
    }
}

extern "C" void custom_max_dim_launcher(const float* x,
                                        float* max_values, int64_t* max_indices,
                                        int B, int C, int H, int W,
                                        int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                        int64_t out_val_s0, int64_t out_val_s1, int64_t out_val_s2, int64_t out_val_s3,
                                        int64_t out_idx_s0, int64_t out_idx_s1, int64_t out_idx_s2, int64_t out_idx_s3,
                                        cudaStream_t stream) {
    int64_t bhw_total = (int64_t)B * H * W;
    int threads = 256;
    int blocks = (int)((bhw_total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_any_reduce_kernel<<<blocks, threads, 0, stream>>>(x, max_values, max_indices, nullptr,
                                                          B, C, H, W,
                                                          x_s0, x_s1, x_s2, x_s3,
                                                          out_val_s0, out_val_s1, out_val_s2, out_val_s3,
                                                          out_idx_s0, out_idx_s1, out_idx_s2, out_idx_s3,
                                                          0, 0, 0, 0, 0.0f);
    CUDA_CHECK_LAUNCH();
}

extern "C" void custom_any_gt_launcher(const float* x,
                                       bool* any_mask,
                                       int B, int C, int H, int W,
                                       int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                       int64_t out_any_s0, int64_t out_any_s1, int64_t out_any_s2, int64_t out_any_s3,
                                       double any_threshold,
                                       cudaStream_t stream) {
    int64_t bhw_total = (int64_t)B * H * W;
    int threads = 256;
    int blocks = (int)((bhw_total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_any_reduce_kernel<<<blocks, threads, 0, stream>>>(x, nullptr, nullptr, any_mask,
                                                          B, C, H, W,
                                                          x_s0, x_s1, x_s2, x_s3,
                                                          0, 0, 0, 0,
                                                          0, 0, 0, 0,
                                                          out_any_s0, out_any_s1, out_any_s2, out_any_s3,
                                                          (float)any_threshold);
    CUDA_CHECK_LAUNCH();
}

__global__ void threshold_mask_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      int B, int C, int H, int W,
                                      int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                      int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3,
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
            int64_t x_idx = b * x_s0 + c * x_s1 + h * x_s2 + w * x_s3;
            float val = x[x_idx];
            if (val > max_val) max_val = val;
            if (val > any_threshold) any_gt = true;
        }
        bool mask = (max_val > max_threshold) && any_gt;
        if (mask) {
            for (int c = 0; c < C; ++c) {
                int64_t x_idx = b * x_s0 + c * x_s1 + h * x_s2 + w * x_s3;
                int64_t out_idx = b * out_s0 + c * out_s1 + h * out_s2 + w * out_s3;
                out[out_idx] = x[x_idx];
            }
        } else {
            for (int c = 0; c < C; ++c) {
                int64_t out_idx = b * out_s0 + c * out_s1 + h * out_s2 + w * out_s3;
                out[out_idx] = 0.0f;
            }
        }
    }
}

extern "C" void custom_threshold_mask_launcher(const float* x, float* out,
                                                int B, int C, int H, int W,
                                                int64_t x_s0, int64_t x_s1, int64_t x_s2, int64_t x_s3,
                                                int64_t out_s0, int64_t out_s1, int64_t out_s2, int64_t out_s3,
                                                double max_threshold, double any_threshold,
                                                cudaStream_t stream) {
    int64_t bhw_total = (int64_t)B * H * W;
    int threads = 256;
    int blocks = (int)((bhw_total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    threshold_mask_kernel<<<blocks, threads, 0, stream>>>(x, out, B, C, H, W,
                                                          x_s0, x_s1, x_s2, x_s3,
                                                          out_s0, out_s1, out_s2, out_s3,
                                                          (float)max_threshold, (float)any_threshold);
    CUDA_CHECK_LAUNCH();
}
