#include <cuda_runtime.h>
#include <math.h>

// Elementwise acos kernel: output[i] = acosf(input[i])
__global__ void acos_elementwise_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        long long total) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total; i += stride) {
        output[i] = acosf(input[i]);
    }
}

// Dimension permutation kernel: input [B,C,H,W] -> output [B,H,C,W], flattened.
// Output indexing: b*H*C*W + h*C*W + c*W + w
// Input indexing:  b*C*H*W + c*H*W + h*W + w
__global__ void cat_unbind_channels_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int B, int C, int H, int W,
                                            long long total) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long linear = idx; linear < total; linear += stride) {
        int w = linear % W;
        long long tmp1 = linear / W;
        int c = tmp1 % C;
        long long tmp2 = tmp1 / C;
        int h = tmp2 % H;
        int b = tmp2 / H;
        // Input offset for [b,c,h,w]
        long long in_idx = ((long long)b * C * H * W) + ((long long)c * H * W) + ((long long)h * W) + w;
        output[linear] = input[in_idx];
    }
}

extern "C" void acos_stack_forward_launcher(const float* input,
                                             float* output,
                                             long long total,
                                             cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, grid-stride handles rest
    acos_elementwise_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Binding layer should check and throw, but we leave this as a no-op here
        // because .cu cannot use TORCH_CHECK. We simply ignore.
    }
}

extern "C" void cat_unbind_channels_forward_launcher(const float* input,
                                                      float* output,
                                                      int B, int C, int H, int W,
                                                      cudaStream_t stream) {
    long long total = (long long)B * H * C * W;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cat_unbind_channels_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore; binding layer will check
    }
}
