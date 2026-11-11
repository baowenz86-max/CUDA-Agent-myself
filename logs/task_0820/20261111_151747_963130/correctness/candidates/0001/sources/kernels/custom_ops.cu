#include <cuda_runtime.h>
#include <cmath>

// Kernel for bilinear upsampling (align_corners=false, scale_factor=2.0)
// Input: (N, C, H_in, W_in) float32
// Output: (N, C, H_out, W_out) float32
// Each thread computes one output pixel.
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_h, float scale_w,
    bool align_corners)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Decompose linear index
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    // Compute source coordinates
    float h_src, w_src;
    if (align_corners) {
        if (H_out > 1)
            h_src = h_out * (H_in - 1) / (float)(H_out - 1);
        else
            h_src = 0.0f;
        if (W_out > 1)
            w_src = w_out * (W_in - 1) / (float)(W_out - 1);
        else
            w_src = 0.0f;
    } else {
        // PyTorch formula for align_corners=False:
        // src = (dst + 0.5) * scale - 0.5, where scale = in/out
        // Here scale_h = H_in / H_out, scale_w = W_in / W_out
        h_src = (h_out + 0.5f) * scale_h - 0.5f;
        w_src = (w_out + 0.5f) * scale_w - 0.5f;
    }

    // Clamp source coordinates to valid range [0, H_in-1], [0, W_in-1]
    // PyTorch clamps to [-0.5, size-0.5]? Actually it clamps to [0, size-1] after floor.
    // We'll handle boundaries by clamping indices directly and using weight computation.
    float h_low_f = floorf(h_src);
    float w_low_f = floorf(w_src);
    int h_low = (int)h_low_f;
    int w_low = (int)w_low_f;
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    // Clamp indices to valid range
    if (h_low < 0) h_low = 0;
    if (h_low >= H_in) h_low = H_in - 1;
    if (w_low < 0) w_low = 0;
    if (w_low >= W_in) w_low = W_in - 1;
    if (h_high < 0) h_high = 0;
    if (h_high >= H_in) h_high = H_in - 1;
    if (w_high < 0) w_high = 0;
    if (w_high >= W_in) w_high = W_in - 1;

    // Interpolation weights
    float h_lambda = h_src - h_low_f;
    float w_lambda = w_src - w_low_f;

    // Input base offset for this n,c
    const float* in_nc = input + ((long long)n * C + c) * H_in * W_in;

    // Bilinear interpolation
    float v00 = in_nc[h_low * W_in + w_low];
    float v01 = in_nc[h_low * W_in + w_high];
    float v10 = in_nc[h_high * W_in + w_low];
    float v11 = in_nc[h_high * W_in + w_high];

    float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
    float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
    float val = top * (1.0f - h_lambda) + bottom * h_lambda;

    // Write output
    long long out_idx = ((long long)n * C + c) * H_out * W_out + (long long)h_out * W_out + w_out;
    output[out_idx] = val;
}

// Kernel for element-wise ceiling
__global__ void ceil_tensor_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    output[idx] = ceilf(input[idx]);
}

// Launcher for bilinear upsampling
// Input: x (float*, contiguous, shape [N,C,H_in,W_in])
// Output: out (float*, allocated shape [N,C,H_out,W_out])
// scale_factor: output/input ratio (as float), e.g. 2.0
// align_corners: bool
// recompute_scale_factor: ignored, but present to match signature
// stream: CUDA stream
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    double scale_factor,
    bool align_corners,
    bool recompute_scale_factor,
    cudaStream_t stream)
{
    // Compute output spatial dimensions
    int H_out = (int)(H_in * scale_factor);
    int W_out = (int)(W_in * scale_factor);

    // scale for align_corners=false = in/out
    float scale_h = (float)H_in / H_out;
    float scale_w = (float)W_in / W_out;

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Launch kernel
    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H_in, W_in,
        H_out, W_out,
        scale_h, scale_w,
        align_corners
    );
}

// Launcher for ceil_tensor
// Input: x (float*, contiguous, numel elements)
// Output: out (float*, allocated same shape)
// stream: CUDA stream
extern "C" void ceil_tensor_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 2147483647) blocks_ll = 2147483647; // grid dim limit
    int blocks = (int)blocks_ll;

    // Use grid-stride loop if needed to handle large numel beyond grid limit
    if ((long long)blocks * threads < numel) {
        // fallback to grid-stride loop kernel
        int grid = blocks;
        ceil_tensor_kernel_grid_stride<<<grid, threads, 0, stream>>>(input, output, numel);
    } else {
        ceil_tensor_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
    }
}

// Grid-stride version of ceil_tensor for very large arrays
__global__ void ceil_tensor_kernel_grid_stride(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        output[i] = ceilf(input[i]);
    }
}
