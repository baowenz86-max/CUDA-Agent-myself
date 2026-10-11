#include <cuda_runtime.h>

// Fused kernel: reflection padding + histogram (float32)
// Input is contiguous NCHW tensor. Output is histogram of size bins.
// Each thread processes one element of the original (unpadded) tensor.
// It computes reflected source indices and atomically increments the appropriate bin.
__global__ void reflection_pad_histc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int pad,
    int bins,
    float min_val,
    float max_val,
    float bin_width)
{
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose linear index into n,c,h,w
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // Compute reflected source coordinates for padding
    int src_h = h - pad;
    if (src_h < 0) src_h = -src_h;
    else if (src_h >= H) src_h = 2 * H - 2 - src_h;

    int src_w = w - pad;
    if (src_w < 0) src_w = -src_w;
    else if (src_w >= W) src_w = 2 * W - 2 - src_w;

    // Safety clamp (should not be needed if pad < H and pad < W, but ensures no OOB)
    src_h = min(max(src_h, 0), H - 1);
    src_w = min(max(src_w, 0), W - 1);

    // Load value from reflected coordinate
    float val = input[((n * C + c) * H + src_h) * W + src_w];

    // Clamp to [min_val, max_val]
    val = fminf(fmaxf(val, min_val), max_val);

    // Compute bin index using the same method as PyTorch's histc:
    // bin = (int)((val - min_val) * (bins / (max_val - min_val)));
    // then handle max_val edge case.
    int bin = (int)((val - min_val) * (bins / (max_val - min_val)));
    if (bin >= bins) bin = bins - 1;
    if (bin < 0) bin = 0;

    // Atomic increment histogram bin
    atomicAdd(&output[bin], 1.0f);
}

// Host launcher with C linkage
extern "C" void reflection_pad_histc_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad,
    int bins,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    if (bins <= 0) return;
    float bin_width = (max_val - min_val) / bins; // Kept for possible future use, but not used in kernel

    int total = N * C * H * W;
    if (total == 0) return;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    reflection_pad_histc_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, pad, bins, min_val, max_val, bin_width);
}
