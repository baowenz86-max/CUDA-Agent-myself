#include <cuda_runtime.h>

// Constant padding kernel: fills output with value, and copies input to appropriate interior region
// Assumes input and output are contiguous 4D tensors with dimensions N,C,H,W and padded output N,C,H+top+bottom, W+left+right
// padding: left, right, top, bottom
__global__ void constant_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int left, int right, int top, int bottom,
    float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Compute output coordinates
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int oc = (i / (outW * outH)) % C;
        int on = i / (outW * outH * C);

        // Check if inside original region
        if (oh >= top && oh < top + H && ow >= left && ow < left + W) {
            int ih = oh - top;
            int iw = ow - left;
            int input_idx = ((on * C + oc) * H + ih) * W + iw;
            output[i] = input[input_idx];
        } else {
            output[i] = value;
        }
    }
}

// Reflection padding kernel: output dimensions outH = H + top + bottom, outW = W + left + right
// Reflection without repeating edge: input index maps as i -> reflect(i)
__device__ __forceinline__ int reflect_index(int idx, int len) {
    // len > 1 always for valid inputs; for safety handle len==1
    if (len == 1) return 0;
    int period = 2 * (len - 1);
    int mod = idx % period;
    if (mod < 0) mod += period;
    return (mod < len) ? mod : (period - mod);
}

__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int left, int right, int top, int bottom)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int oc = (i / (outW * outH)) % C;
        int on = i / (outW * outH * C);

        // Map output position to input position using reflection
        int ih = reflect_index(oh - top, H);
        int iw = reflect_index(ow - left, W);
        int input_idx = ((on * C + oc) * H + ih) * W + iw;
        output[i] = input[input_idx];
    }
}

// Launcher for constant_pad2d
// Input and output pointers must be CUDA pointers (device memory)
extern "C" void constant_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int left, int right, int top, int bottom,
    float value,
    cudaStream_t stream)
{
    int outH = H + top + bottom;
    int outW = W + left + right;
    int total = N * C * outH * outW;

    // Choose block size and grid size (grid-stride loop covers all)
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to a reasonable number for grid-stride loop overhead, but ensure enough
    if (blocks > 65535) blocks = 65535;
    // Minimum 1 block for non-empty output (empty handled by binding)
    if (blocks < 1) blocks = 1;

    constant_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, left, right, top, bottom, value);
}

// Launcher for reflection_pad2d
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int left, int right, int top, int bottom,
    cudaStream_t stream)
{
    int outH = H + top + bottom;
    int outW = W + left + right;
    int total = N * C * outH * outW;

    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, left, right, top, bottom);
}
