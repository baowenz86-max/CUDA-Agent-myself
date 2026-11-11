// CUDA kernels for upsample_bilinear2x_forward and rot90_forward
// Only supports float32 on CUDA. No PyTorch dependencies.
#include <cuda_runtime.h>
#include <cstdint>

// Helper to get linear index for 4D tensor given strides
__device__ __forceinline__ int64_t get_index(int64_t n, int64_t c, int64_t h, int64_t w,
                                             int64_t sN, int64_t sC, int64_t sH, int64_t sW) {
    return n * sN + c * sC + h * sH + w * sW;
}

// Bilinear upsampling with scale_factor=2, align_corners=False
// Input: (N, C, H, W), output: (N, C, 2*H, 2*W)
// input_strides: sN, sC, sH, sW (in elements, int64_t)
// output assumed contiguous with strides (C*2H*2W, 2H*2W, 2W, 1)
__global__ void upsample_bilinear2x_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int64_t sN, int64_t sC, int64_t sH, int64_t sW,
    int outH, int outW)
{
    // outH = 2*H, outW = 2*W
    int64_t total = (int64_t)N * C * outH * outW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // decompose idx
        int w_out = idx % outW;
        int64_t tmp = idx / outW;
        int h_out = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // source coordinate mapping for align_corners=False, scale=2
        float src_h = (h_out + 0.5f) * 0.5f - 0.5f;
        float src_w = (w_out + 0.5f) * 0.5f - 0.5f;
        src_h = fmaxf(0.0f, src_h);
        src_w = fmaxf(0.0f, src_w);
        int h0 = (int)src_h;
        int w0 = (int)src_w;
        int h1 = min(h0 + 1, H - 1);
        int w1 = min(w0 + 1, W - 1);
        float h_lambda = src_h - h0;
        float w_lambda = src_w - w0;

        // fetch four values using input strides
        int64_t base00 = get_index(n, c, h0, w0, sN, sC, sH, sW);
        int64_t base01 = get_index(n, c, h0, w1, sN, sC, sH, sW);
        int64_t base10 = get_index(n, c, h1, w0, sN, sC, sH, sW);
        int64_t base11 = get_index(n, c, h1, w1, sN, sC, sH, sW);

        float v00 = input[base00];
        float v01 = input[base01];
        float v10 = input[base10];
        float v11 = input[base11];

        float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
        float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
        float val = top * (1.0f - h_lambda) + bottom * h_lambda;

        output[idx] = val;
    }
}

// rot90 with k=1, dims=(2,3) (swap last two dims and flip new last dim)
// Input: (N, C, H, W), output: (N, C, W, H) for k=1? Actually rot90 k=1 dims=(2,3) rotates 90 CCW:
// For a matrix (H,W), output shape becomes (W,H). The formula: out[n,c,j,i] = in[n,c,i, W-1-j]
// Here input is (N,C,H,W), output is (N,C,W,H) where out_w = H, out_h = W.
// output is contiguous with strides (C*W*H, W*H, H, 1) where out shape is [N,C,W,H]? 
// Actually output shape after rot90: dims 2 and 3 swapped -> [N,C,W,H]. So output H_out = W, W_out = H.
// Using index decomposition of output linear index (contiguous)
__global__ void rot90_k_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int64_t sN, int64_t sC, int64_t sH, int64_t sW)
{
    // output shape: (N, C, W, H) ; outH = W, outW = H
    int outH = W;
    int outW = H;
    int64_t total = (int64_t)N * C * outH * outW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int w_out = idx % outW;   // corresponds to input H index? Actually output coordinate (j,i): j varies along W_out=H, i along H_out=W.
        int64_t tmp = idx / outW;
        int h_out = tmp % outH;   // h_out index corresponds to j (0..W-1)
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // For rot90 k=1 dims=(2,3): out[n,c,j,i] = in[n,c,i, W-1-j]
        int in_h = w_out;          // i -> H dimension of input (0..H-1)
        int in_w = W - 1 - h_out;  // j -> W dimension of input flipped

        int64_t in_idx = get_index(n, c, in_h, in_w, sN, sC, sH, sW);
        output[idx] = input[in_idx];
    }
}

// Generic rot90 for any k (0,1,2,3) with dims=(2,3)
__global__ void rot90_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int64_t sN, int64_t sC, int64_t sH, int64_t sW,
    int k)
{
    int kmod = ((k % 4) + 4) % 4;
    int outH, outW;
    if (kmod == 0 || kmod == 2) {
        outH = H; outW = W;
    } else {
        outH = W; outW = H;
    }
    int64_t total = (int64_t)N * C * outH * outW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int w_out = idx % outW;
        int64_t tmp = idx / outW;
        int h_out = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        int in_h, in_w;
        if (kmod == 0) {
            in_h = h_out; in_w = w_out;
        } else if (kmod == 1) {
            in_h = w_out; in_w = W - 1 - h_out;
        } else if (kmod == 2) {
            in_h = H - 1 - h_out; in_w = W - 1 - w_out;
        } else { // kmod == 3
            in_h = H - 1 - w_out; in_w = h_out;
        }

        int64_t in_idx = get_index(n, c, in_h, in_w, sN, sC, sH, sW);
        output[idx] = input[in_idx];
    }
}

// Host launcher for upsample_bilinear2x
// Assumes input and output are CUDA float pointers, input may be non-contiguous with given strides (elements).
// output is contiguous with shape [N, C, 2H, 2W].
extern "C" void upsample_bilinear2x_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream)
{
    int outH = 2 * H;
    int outW = 2 * W;
    int64_t total = (int64_t)N * C * outH * outW;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // grid-stride loop ensures coverage, but cap blocks to avoid excessive launch overhead
    int maxBlocks = 65535;
    if (blocks > maxBlocks) blocks = maxBlocks;
    upsample_bilinear2x_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        outH, outW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot throw, but we can leave error to be checked by binding layer via cudaGetLastError
        // We just return.
    }
}

// Host launcher for rot90 with dims fixed (2,3). k is expected to be int.
extern "C" void rot90_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int k,
    cudaStream_t stream)
{
    int64_t total;
    int outH, outW;
    int kmod = ((k % 4) + 4) % 4;
    if (kmod == 0 || kmod == 2) {
        outH = H; outW = W;
    } else {
        outH = W; outW = H;
    }
    total = (int64_t)N * C * outH * outW;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    int maxBlocks = 65535;
    if (blocks > maxBlocks) blocks = maxBlocks;
    // Use generic kernel to handle any k, but for k=1 we could use specialized; generic is fine.
    rot90_generic_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignored
    }
}
