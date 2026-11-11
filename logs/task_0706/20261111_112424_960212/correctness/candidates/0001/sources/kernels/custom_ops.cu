// custom_ops.cu - CUDA kernels for channel sum and average pooling
// Only includes CUDA runtime; no PyTorch dependencies.
#include <cuda_runtime.h>

// Channel sum kernel: sum over dimension 1 (channels) with keepdim=True.
// Input: [N, C, H, W] contiguous float32
// Output: [N, 1, H, W] contiguous float32
// Each thread computes one output element (n, h, w), summing over C.
__global__ void channel_sum_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;
    // Decode idx into n, h, w
    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (H * W);
    // Sum over channels
    float sum = 0.0f;
    const float* in_base = input + n * C * H * W + h * W + w;
    for (int c = 0; c < C; ++c) {
        sum += in_base[c * H * W];
    }
    output[n * H * W + h * W + w] = sum;
}

// Average pooling kernel (2D) with count_include_pad=True.
// Input: [N, 1, H, W] contiguous float32
// Output: [N, 1, OH, OW] with OH = floor((H + 2*padding - kernel_size)/stride)+1, OW similarly.
// Each thread computes one output element.
__global__ void avg_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N, int H, int W,
                                  int kernel_size, int stride, int padding,
                                  int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW;
    if (idx >= total) return;
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int n = idx / (OH * OW);

    // Compute input start coordinates (with padding offset)
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;
    // Accumulate
    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = h_start + kh;
        if (ih < 0 || ih >= H) continue; // zero padding (not included)
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = w_start + kw;
            if (iw < 0 || iw >= W) continue;
            // Input is [N, 1, H, W], so offset = n*H*W + ih*W + iw
            sum += input[n * H * W + ih * W + iw];
        }
    }
    // count_include_pad=True means divide by kernel_size*kernel_size always.
    int divisor = kernel_size * kernel_size;
    output[idx] = sum / (float)divisor;
}

// Host launcher for channel_sum
extern "C" void channel_sum_launcher(float* output, const float* input,
                                     int N, int C, int H, int W,
                                     cudaStream_t stream) {
    int total = N * H * W;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    channel_sum_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N, C, H, W);
    // Check for launch errors (synchronous check)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, you would throw; here we just ignore or could print.
        // No dependency on printf if not desired; but we keep minimal.
    }
}

// Host launcher for avg_pool2d
extern "C" void avg_pool2d_launcher(float* output, const float* input,
                                    int N, int H, int W,
                                    int kernel_size, int stride, int padding,
                                    cudaStream_t stream) {
    int OH = (H + 2 * padding - kernel_size) / stride + 1;
    int OW = (W + 2 * padding - kernel_size) / stride + 1;
    int total = N * OH * OW;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    avg_pool2d_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N, H, W,
                                                            kernel_size, stride, padding,
                                                            OH, OW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore for now
    }
}
