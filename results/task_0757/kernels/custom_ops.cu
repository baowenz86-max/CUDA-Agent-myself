#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// custom_cat: concatenate two float32 tensors along dimension 1 (dim=1).
// Input tensors may be non-contiguous; use their strides to index.
// Output tensor is contiguous with merged shape.
// -----------------------------------------------------------------------------
__global__ void custom_cat_kernel(
    float* __restrict__ output,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    int64_t out_numel,
    int N,      // batch
    int C1,     // channels of x1
    int C2,     // channels of x2
    int H,
    int W,
    // strides for x1 (4 dims)
    int64_t s1_N, int64_t s1_C, int64_t s1_H, int64_t s1_W,
    // strides for x2 (4 dims)
    int64_t s2_N, int64_t s2_C, int64_t s2_H, int64_t s2_W
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < out_numel; i += stride) {
        // output has shape [N, C1+C2, H, W], contiguous
        int64_t w = i % W;
        int64_t tmp = i / W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % (C1 + C2);
        int64_t n = tmp / (C1 + C2);

        const float* src;
        int64_t src_index;
        if (c < C1) {
            src = x1;
            src_index = n * s1_N + c * s1_C + h * s1_H + w * s1_W;
        } else {
            src = x2;
            int64_t c2 = c - C1;
            src_index = n * s2_N + c2 * s2_C + h * s2_H + w * s2_W;
        }
        output[i] = src[src_index];
    }
}

extern "C" void custom_cat_launcher(
    float* output,
    const float* x1,
    const float* x2,
    int64_t out_numel,
    int N,
    int C1,
    int C2,
    int H,
    int W,
    int64_t s1_N, int64_t s1_C, int64_t s1_H, int64_t s1_W,
    int64_t s2_N, int64_t s2_C, int64_t s2_H, int64_t s2_W,
    cudaStream_t stream
) {
    const int threads = 256;
    int blocks = (int)((out_numel + threads - 1) / threads);
    // Cap blocks to avoid excessive grid size; grid-stride loop handles rest
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;

    custom_cat_kernel<<<blocks, threads, 0, stream>>>(
        output, x1, x2, out_numel,
        N, C1, C2, H, W,
        s1_N, s1_C, s1_H, s1_W,
        s2_N, s2_C, s2_H, s2_W
    );
    cudaGetLastError();
}

// -----------------------------------------------------------------------------
// custom_ne: element-wise not-equal to scalar, output is bool.
// Input float32, output bool (C++ bool), output is contiguous.
// -----------------------------------------------------------------------------
__global__ void custom_ne_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    int64_t numel,
    float other
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] != other);
    }
}

extern "C" void custom_ne_launcher(
    const float* input,
    bool* output,
    int64_t numel,
    float other,
    cudaStream_t stream
) {
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;

    custom_ne_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel, other
    );
    cudaGetLastError();
}
