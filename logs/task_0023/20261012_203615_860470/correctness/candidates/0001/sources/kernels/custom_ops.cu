#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper: bitonic sort for 128 elements in registers
__device__ void bitonic_sort_128(float* arr) {
    // Stage 1: sort pairs
    for (int k = 2; k <= 128; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int i = 0; i < 128; ++i) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    float a = arr[i];
                    float b = arr[ixj];
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        arr[i] = b;
                        arr[ixj] = a;
                    }
                }
            }
        }
    }
}

// Kernel: median along dim0 for float32, shape [N,C,H,W]
// Output shape [C,H,W], lower median for even N
__global__ void median_dim0_kernel(const float* input, float* output,
                                   int N, int C, int H, int W) {
    int spatial = H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;

    int c = idx / spatial;
    int hw = idx % spatial;

    float vals[128];
    // Load values along N dimension; input is contiguous with shape [N,C,H,W]
    const float* base = input + c * spatial + hw;
    #pragma unroll
    for (int n = 0; n < 128; ++n) {
        vals[n] = base[n * C * spatial]; // stride between n is C*spatial
    }

    bitonic_sort_128(vals);

    // Lower median: index (N-1)/2 for even N, N/2 for odd (but N=128 even)
    output[idx] = vals[(N - 1) / 2];
}

// Kernel: reflection padding 2D for each channel slice
// input shape [C,H,W], output shape [C, H+2*pad, W+2*pad]
__global__ void reflection_pad2d_kernel(const float* input, float* output,
                                        int C, int H, int W, int pad) {
    int outH = H + 2 * pad;
    int outW = W + 2 * pad;
    int total = C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % outW;
    int tmp = idx / outW;
    int oh = tmp % outH;
    int c = tmp / outH;

    // Compute reflected coordinates
    int ih = oh - pad;
    if (ih < 0) ih = -ih;
    if (ih >= H) ih = 2 * (H - 1) - ih;

    int iw = ow - pad;
    if (iw < 0) iw = -iw;
    if (iw >= W) iw = 2 * (W - 1) - iw;

    output[idx] = input[(c * H + ih) * W + iw];
}

extern "C" void median_dim0_forward_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {

    int spatial = H * W;
    int total = C * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    median_dim0_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
}

extern "C" void reflection_pad2d_forward_launcher(
    const float* input, float* output,
    int C, int H, int W, int pad,
    cudaStream_t stream) {

    int outH = H + 2 * pad;
    int outW = W + 2 * pad;
    int total = C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(input, output, C, H, W, pad);
}
