#include <cuda_runtime.h>

// Kernel for element-wise absolute value
// Supports arbitrary strides; each thread handles one element via grid-stride loop
__global__ void abs_kernel(float* output, const float* input, long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = fabsf(input[idx]);
    }
}

// Kernel for argmin along dimension 1.
// Input shape: [B, D, H, W] (dim=1)
// Output shape: [B, H, W]
// Each thread computes argmin for one (b, h, w).
// Input and output are assumed contiguous in memory layout, but we use explicit strides
// to be safe. The strides are passed as arguments.
__global__ void argmin_kernel(const float* input, int64_t* output,
                              int B, int D, int H, int W,
                              long long in_stride_b, long long in_stride_d,
                              long long in_stride_h, long long in_stride_w,
                              long long out_stride_b, long long out_stride_h,
                              long long out_stride_w) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)B * H * W;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int b = idx / ((long long)H * W);
        int rem = idx % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        const float* base = input + b * in_stride_b + h * in_stride_h + w * in_stride_w;
        float min_val = base[0];
        int min_idx = 0;
        for (int d = 1; d < D; ++d) {
            float val = base[d * in_stride_d];
            if (val < min_val) {
                min_val = val;
                min_idx = d;
            }
        }
        output[b * out_stride_b + h * out_stride_h + w * out_stride_w] = (int64_t)min_idx;
    }
}

// Launcher for abs kernel
extern "C" void copy_tensor_k_abs_out_launcher(float* output, const float* input,
                                                long long numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;  // grid-stride loop handles the rest
    abs_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

// Launcher for argmin kernel
extern "C" void copy_tensor_k_argmin_out_launcher(const float* input, int64_t* output,
                                                   int B, int D, int H, int W,
                                                   long long in_stride_b, long long in_stride_d,
                                                   long long in_stride_h, long long in_stride_w,
                                                   long long out_stride_b, long long out_stride_h,
                                                   long long out_stride_w,
                                                   cudaStream_t stream) {
    long long total = (long long)B * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    argmin_kernel<<<blocks, threads, 0, stream>>>(input, output, B, D, H, W,
                                                  in_stride_b, in_stride_d,
                                                  in_stride_h, in_stride_w,
                                                  out_stride_b, out_stride_h,
                                                  out_stride_w);
}
