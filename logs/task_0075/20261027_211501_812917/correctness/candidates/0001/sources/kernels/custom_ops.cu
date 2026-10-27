// CUDA kernels for reflection_pad2d, count_nonzero, upsample_bilinear2d
// All functions use raw pointers and cudaStream_t, no torch/ATen dependencies.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Helper to check CUDA errors after kernel launch (only in debug or always? we do always for safety)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// 1. ReflectionPad2d
// -----------------------------------------------------------------------------
//
// Applies reflection padding to a 4D tensor (N, C, H, W).
// Padding is symmetric: same pad value on all four spatial sides.
// Supports arbitrary strides by using the input tensor's strides (in elements).
// Output is contiguous.

__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int pad,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    long long total_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total_elements; idx += stride) {
        // Decompose linear index into output coordinates: n, c, oh, ow
        int ow = idx % outW;
        long long tmp = idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // Reflect indices: for pad=1, output indices 0 and H+pad-1 are mapped to 1 and H-2, etc.
        // General reflection:
        // if (i < pad)          ih = pad - i;
        // else if (i >= H + pad) ih = (H - 1) - (i - H - pad + 1) = 2*H - i + 2*pad - 2? careful.
        // Simpler: map output spatial index 0..outH-1 to input index 0..H-1 using reflection without repeating edge.
        // Use formula: ih = reflect(oh, pad, H)
        int ih = oh - pad;
        if (ih < 0) ih = -ih;
        else if (ih >= H) ih = 2 * H - ih - 2;

        int iw = ow - pad;
        if (iw < 0) iw = -iw;
        else if (iw >= W) iw = 2 * W - iw - 2;

        // Compute input linear index using strides (in elements)
        long long in_idx = (long long)n * in_n_stride + (long long)c * in_c_stride
                        + (long long)ih * in_h_stride + (long long)iw * in_w_stride;
        output[idx] = input[in_idx];
    }
}

extern "C" void reflection_pad2d_forward_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    int pad,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    cudaStream_t stream)
{
    int outH = H + 2 * pad;
    int outW = W + 2 * pad;
    long long total_elements = (long long)N * C * outH * outW;

    // Launch configuration: 256 threads per block, enough blocks to cover grid or cap at 65535*256? use grid-stride loop anyway.
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    // Cap blocks to reasonable max grid dimension (e.g., 65535) but our loop handles rest; however need at least 1 block.
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, pad,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        total_elements);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// 2. CountNonzero
// -----------------------------------------------------------------------------
// Counts number of non-zero elements (including NaN, since NaN != 0) across entire tensor.
// Returns int64 scalar in output[0].
// We use a two-pass approach: first each block counts nonzeros and writes partial sum to a temporary device buffer.
// But launcher only gets output pointer; we need temp storage? We can use atomicAdd directly on output[0] (int64_t atomicAdd is supported on sm_60+).
// For simplicity and since target is modern GPUs (sm_80), we use atomicAdd on unsigned long long, casting count. Ensure output initialized to zero before kernel.
// But binding layer allocates output tensor and should zero it. We'll also zero it in kernel? Not safe if multiple blocks write. Better assume caller zeroes output. We'll instruct in contract? But we can also use a separate temp buffer allocated via cudaMalloc? Cannot allocate in launcher (we can, but no torch). However, binding.cpp will allocate a zero-initialized int64 scalar and pass pointer. We'll document that the binding must zero initialize. Alternatively, we can zero it here using cudaMemsetAsync before kernel launch, but that is safe and no torch dependency. We'll do cudaMemsetAsync to zero output[0] before kernel. Since output is int64_t, we need to memset sizeof(int64_t)=8 bytes.

__global__ void count_nonzero_kernel(
    const float* __restrict__ input,
    long long total_elements,
    int64_t* __restrict__ output_count)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    long long local_count = 0;

    for (; idx < total_elements; idx += stride) {
        if (input[idx] != 0.0f) {
            local_count++;
        }
    }

    // Atomic add to global count (int64_t atomicAdd is available on sm_60+; we assume target >= 8.0)
    atomicAdd((unsigned long long*)output_count, (unsigned long long)local_count);
}

extern "C" void count_nonzero_forward_launcher(
    const float* input,
    long long total_elements,
    int64_t* output_count,
    cudaStream_t stream)
{
    // Zero the output scalar using cudaMemsetAsync (async on stream)
    cudaMemsetAsync(output_count, 0, sizeof(int64_t), stream);

    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;

    count_nonzero_kernel<<<blocks, threads, 0, stream>>>(
        input, total_elements, output_count);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// 3. UpsampleBilinear2d (scale_factor float, align_corners bool)
// -----------------------------------------------------------------------------
// Performs bilinear upsampling with PyTorch semantics: align_corners=False default.
// Output size determined by floor(input_size * scale_factor). For integer scale_factor (2.0) we can compute directly.
// Kernel assumes output tensor is contiguous. Input may have arbitrary strides; we use strides in elements.

__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_factor,
    bool align_corners,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    long long total_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    // Precompute scaling parameters
    float rheight = (outH > 1) ? (align_corners ? (float)(H - 1) / (outH - 1) : 1.0f / scale_factor) : 0.0f;
    float rwidth = (outW > 1) ? (align_corners ? (float)(W - 1) / (outW - 1) : 1.0f / scale_factor) : 0.0f;

    for (; idx < total_elements; idx += stride) {
        int ow = idx % outW;
        long long tmp = idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // Source coordinates for align_corners=False (PyTorch default):
        // src = (dst + 0.5) / scale - 0.5
        float h1r, w1r;
        if (align_corners) {
            h1r = oh * rheight;
            w1r = ow * rwidth;
        } else {
            h1r = (oh + 0.5f) * rheight - 0.5f;
            w1r = (ow + 0.5f) * rwidth - 0.5f;
        }

        // Clamp to [0, H-1] and [0, W-1]
        h1r = fmaxf(0.0f, fminf(h1r, (float)(H - 1)));
        w1r = fmaxf(0.0f, fminf(w1r, (float)(W - 1)));

        int h1 = (int)h1r;
        int h1p = (h1 < H - 1) ? h1 + 1 : h1;
        float h1lambda = h1r - h1;
        float h0lambda = 1.0f - h1lambda;

        int w1 = (int)w1r;
        int w1p = (w1 < W - 1) ? w1 + 1 : w1;
        float w1lambda = w1r - w1;
        float w0lambda = 1.0f - w1lambda;

        // Load four values using strides
        long long base = (long long)n * in_n_stride + (long long)c * in_c_stride;
        float v00 = input[base + (long long)h1 * in_h_stride + (long long)w1 * in_w_stride];
        float v01 = input[base + (long long)h1 * in_h_stride + (long long)w1p * in_w_stride];
        float v10 = input[base + (long long)h1p * in_h_stride + (long long)w1 * in_w_stride];
        float v11 = input[base + (long long)h1p * in_h_stride + (long long)w1p * in_w_stride];

        float result = h0lambda * (w0lambda * v00 + w1lambda * v01) +
                       h1lambda * (w0lambda * v10 + w1lambda * v11);

        output[idx] = result;
    }
}

extern "C" void upsample_bilinear2d_forward_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    float scale_factor,
    bool align_corners,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    cudaStream_t stream)
{
    int outH = (int)floorf(H * scale_factor);
    int outW = (int)floorf(W * scale_factor);
    long long total_elements = (long long)N * C * outH * outW;

    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;

    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, scale_factor, align_corners,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride, total_elements);
    CUDA_CHECK(cudaGetLastError());
}
