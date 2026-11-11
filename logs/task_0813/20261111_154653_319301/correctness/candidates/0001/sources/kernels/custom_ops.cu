#include <cuda_runtime.h>

// ceil_forward: element-wise ceil for float32
__global__ void ceil_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = ceilf(input[i]);
    }
}

// reflection_pad2d: output shape [N,C,H+pad_top+pad_bottom, W+pad_left+pad_right]
// input shape [N,C,H,W], contiguous
__global__ void reflection_pad2d_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int N, int C, int H, int W,
                                        int pad_left, int pad_right, int pad_top, int pad_bottom) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t out_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int out_h = H + pad_top + pad_bottom;
    int out_w = W + pad_left + pad_right;
    int64_t out_numel = static_cast<int64_t>(N) * C * out_h * out_w;

    for (int64_t i = out_idx; i < out_numel; i += out_stride) {
        int ow = i % out_w;
        int64_t tmp = i / out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int c = tmp % C;
        int n = tmp / C;

        int iw = ow - pad_left;
        int ih = oh - pad_top;

        // Reflect index in width dimension
        if (iw < 0) {
            iw = -iw;
        }
        if (iw >= W) {
            iw = 2 * W - iw - 2;
        }
        // Reflect index in height dimension
        if (ih < 0) {
            ih = -ih;
        }
        if (ih >= H) {
            ih = 2 * H - ih - 2;
        }
        // Clamp for safety (should be inside bounds for valid padding)
        iw = max(0, min(iw, W - 1));
        ih = max(0, min(ih, H - 1));

        int64_t input_idx = ((static_cast<int64_t>(n) * C + c) * H + ih) * W + iw;
        output[i] = input[input_idx];
    }
}

// Launcher for ceil
extern "C" void ceil_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > 0) {
        // Cap grid size to 2^31-1; kernel uses grid-stride loop
        int grid = static_cast<int>(min<int64_t>(blocks, 2147483647LL));
        ceil_kernel<<<grid, block_size, 0, stream>>>(input, output, numel);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be surfaced by binding layer via cudaGetLastError; do nothing here
    }
}

// Launcher for reflection_pad2d
extern "C" void reflection_pad2d_forward_launcher(const float* input,
                                                   float* output,
                                                   int N, int C, int H, int W,
                                                   int pad_left, int pad_right, int pad_top, int pad_bottom,
                                                   cudaStream_t stream) {
    int out_h = H + pad_top + pad_bottom;
    int out_w = W + pad_left + pad_right;
    int64_t out_numel = static_cast<int64_t>(N) * C * out_h * out_w;
    const int block_size = 256;
    int64_t blocks = (out_numel + block_size - 1) / block_size;
    if (blocks > 0) {
        int grid = static_cast<int>(min<int64_t>(blocks, 2147483647LL));
        reflection_pad2d_kernel<<<grid, block_size, 0, stream>>>(
            input, output, N, C, H, W, pad_left, pad_right, pad_top, pad_bottom);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be surfaced by binding layer via cudaGetLastError; do nothing here
    }
}
