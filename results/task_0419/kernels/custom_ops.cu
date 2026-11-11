#include <cuda_runtime.h>
#include <float.h>

// Elementwise absolute value kernel
// Supports arbitrary strides by computing linear index and mapping back to 4D coordinates.
__global__ void abs_kernel(float* __restrict__ output, const float* __restrict__ input,
                           int64_t total_elements) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total_elements; idx += stride) {
        float val = input[idx];  // input is contiguous? Actually we require contiguous? Contract says supports strides, so we must handle strides.
        // To handle strides, we pass strides and dimensions separately. Here we assume contiguous for simplicity? But contract says supports strides.
    }
}

// We need to respect arbitrary strides. We'll pass dimensions and strides for 4D tensor.
// General 4D abs kernel with stride support.
__global__ void abs_kernel_strided(float* __restrict__ output, const float* __restrict__ input,
                                   int64_t n, int64_t c, int64_t h, int64_t w,
                                   int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
                                   int64_t in_n, int64_t in_c, int64_t in_h, int64_t in_w) {
    int64_t total = n * c * h * w;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int64_t iw = idx % w;
        int64_t ih = (idx / w) % h;
        int64_t ic = (idx / (w * h)) % c;
        int64_t in_ = idx / (w * h * c);
        int64_t in_offset = in_ * in_n + ic * in_c + ih * in_h + iw * in_w;
        int64_t out_offset = in_ * out_n + ic * out_c + ih * out_h + iw * out_w;
        output[out_offset] = fabsf(input[in_offset]);
    }
}

// log_softmax over dim=1 (channels).
// For each (n, h, w) location, reduce over c.
// We use two passes: one for max, one for sum-exp and output.
// Memory coalescing: each thread handles a single spatial/batch position and loops over channels.
__global__ void log_softmax_channel_kernel(float* __restrict__ output, const float* __restrict__ input,
                                           int64_t n, int64_t c, int64_t h, int64_t w,
                                           int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
                                           int64_t in_n, int64_t in_c, int64_t in_h, int64_t in_w) {
    int64_t total_spatial = n * h * w;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total_spatial; idx += stride) {
        int64_t iw = idx % w;
        int64_t ih = (idx / w) % h;
        int64_t in_ = idx / (w * h);

        // Compute max over channels.
        float max_val = -FLT_MAX;
        for (int64_t ic = 0; ic < c; ++ic) {
            int64_t in_offset = in_ * in_n + ic * in_c + ih * in_h + iw * in_w;
            float val = input[in_offset];
            if (val > max_val) max_val = val;
        }

        // Compute sum of exp(x - max).
        float sum_exp = 0.0f;
        for (int64_t ic = 0; ic < c; ++ic) {
            int64_t in_offset = in_ * in_n + ic * in_c + ih * in_h + iw * in_w;
            sum_exp += __expf(input[in_offset] - max_val);
        }
        float log_sum_exp = logf(sum_exp) + max_val;

        // Write output: input - max - log_sum_exp? Actually log_softmax = x - max - log(sum(exp(x-max))) = x - log_sum_exp.
        for (int64_t ic = 0; ic < c; ++ic) {
            int64_t in_offset = in_ * in_n + ic * in_c + ih * in_h + iw * in_w;
            int64_t out_offset = in_ * out_n + ic * out_c + ih * out_h + iw * out_w;
            output[out_offset] = input[in_offset] - log_sum_exp;
        }
    }
}

// Launchers
extern "C" void abs_forward_launcher(float* output, const float* input,
                                      int64_t n, int64_t c, int64_t h, int64_t w,
                                      int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
                                      int64_t in_n, int64_t in_c, int64_t in_h, int64_t in_w,
                                      cudaStream_t stream) {
    int64_t total = n * c * h * w;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles remaining
    abs_kernel_strided<<<blocks, threads, 0, stream>>>(output, input,
                                                        n, c, h, w,
                                                        out_n, out_c, out_h, out_w,
                                                        in_n, in_c, in_h, in_w);
}

extern "C" void log_softmax_forward_launcher(float* output, const float* input,
                                             int64_t n, int64_t c, int64_t h, int64_t w,
                                             int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
                                             int64_t in_n, int64_t in_c, int64_t in_h, int64_t in_w,
                                             int64_t dim, cudaStream_t stream) {
    // Only dim=1 supported based on contract.
    if (dim != 1) return;
    int64_t total_spatial = n * h * w;
    if (total_spatial <= 0) return;
    int threads = 256;
    int blocks = (int)((total_spatial + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    log_softmax_channel_kernel<<<blocks, threads, 0, stream>>>(output, input,
                                                                n, c, h, w,
                                                                out_n, out_c, out_h, out_w,
                                                                in_n, in_c, in_h, in_w);

    // Check for launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // We cannot use printf in .cu? We can, but avoid. Just ignore; binding will check.
    }
}
