#include <cuda_runtime.h>

// CUDA kernel for adaptive average pooling to output_size x output_size
// Each thread computes one output element (n, c, oh, ow)
__global__ void adaptive_avg_pool2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int output_size,
    long input_stride_n, long input_stride_c, long input_stride_h, long input_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * output_size * output_size;
    if (idx >= total) return;

    // Decode indices
    int ow = idx % output_size;
    int oh = (idx / output_size) % output_size;
    int c = (idx / (output_size * output_size)) % C;
    int n = idx / (output_size * output_size * C);

    // Adaptive pooling: compute start and end indices in input for this output cell
    int start_h = (int)floorf((float)oh * H / output_size);
    int end_h = (int)ceilf((float)(oh + 1) * H / output_size);
    int start_w = (int)floorf((float)ow * W / output_size);
    int end_w = (int)ceilf((float)(ow + 1) * W / output_size);

    // Clamp to input bounds (should be fine, but ensure)
    if (start_h < 0) start_h = 0;
    if (end_h > H) end_h = H;
    if (start_w < 0) start_w = 0;
    if (end_w > W) end_w = W;

    float sum = 0.0f;
    int count = 0;
    const float* input_base = input + n * input_stride_n + c * input_stride_c;
    for (int h = start_h; h < end_h; ++h) {
        for (int w = start_w; w < end_w; ++w) {
            sum += input_base[h * input_stride_h + w * input_stride_w];
            ++count;
        }
    }
    output[idx] = sum / (float)count;
}

// Launcher for adaptive_avg_pool2d
// Parameters:
//   output: pointer to output tensor data (float, contiguous)
//   input: pointer to input tensor data (float, possibly non-contiguous)
//   N, C, H, W: input dimensions
//   output_size: target spatial size (2 for our case)
//   input_strides: strides in elements for N, C, H, W dimensions
//   stream: CUDA stream
void adaptive_avg_pool2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int output_size,
    long input_stride_n, long input_stride_c, long input_stride_h, long input_stride_w,
    cudaStream_t stream)
{
    int total = N * C * output_size * output_size;
    if (total == 0) return; // nothing to do

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    adaptive_avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W, output_size,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real launcher, we might want to propagate error, but here we just ignore?
        // Actually cannot throw from extern "C". We rely on binding to check.
        // We can print for debug.
        // printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
