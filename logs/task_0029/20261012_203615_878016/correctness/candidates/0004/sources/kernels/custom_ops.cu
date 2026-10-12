#include <cuda_runtime.h>
#include <cstdint>

// Fused kernel: for each (b,h,w), compute max over C and store value and index.
// Input: [B, C, H, W] float32 contiguous.
// Output values: [B, H, W] float32 contiguous (flatten as B*H*W).
// Output indices: [B, H, W] int64 contiguous.
__global__ void unbind_max_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int B, int C, int H, int W
) {
    // Total number of output elements: B*H*W.
    long long total = (long long)B * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        int b = tmp / H;
        // Input index for (b, 0, h, w)
        long long base = ((long long)b * C * H * W) + (long long)h * W + w;
        float max_val = input[base];
        int64_t max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = input[base + (long long)c * H * W];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        values[idx] = max_val;
        indices[idx] = max_idx;
    }
}

extern "C" void custom_unbind_max_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream
) {
    long long total = (long long)B * H * W;
    const int threads = 256;
    // Choose grid size to cover total with grid-stride loop.
    long long blocks_ll = (total + threads - 1) / threads;
    int blocks = (int)min(blocks_ll, (long long)65535); // gridDim.x max is 65535 on all supported archs
    if (blocks < 1) blocks = 1;
    unbind_max_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, values, indices, B, C, H, W);
}
