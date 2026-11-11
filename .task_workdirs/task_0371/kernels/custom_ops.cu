#include <cuda_runtime.h>

// CrossMapLRN2d forward kernel
// Computes output[n,c,h,w] = input[n,c,h,w] / (k + alpha * sum_{j=max(0,c-size//2)}^{min(C-1,c+size//2)} input[n,j,h,w]^2)^beta
// Supports arbitrary strides (non-contiguous tensors).
// Grid-stride loop over total number of elements (N*C*H*W).
// Input/output are float32.
__global__ void cross_map_lrn_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    long long input_n_stride, long long input_c_stride, long long input_h_stride, long long input_w_stride,
    long long output_n_stride, long long output_c_stride, long long output_h_stride, long long output_w_stride,
    int size,
    float alpha, float beta, float k)
{
    long long total = (long long)N * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_size = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += grid_size) {
        // Decode linear index to n,c,h,w using input strides
        long long tmp = idx;
        int w = tmp % W;
        tmp /= W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        // Compute sum of squares over the channel window
        int half = size / 2;
        int c_start = max(0, c - half);
        int c_end = min(C - 1, c + half);
        float sum_sq = 0.0f;
        for (int j = c_start; j <= c_end; ++j) {
            // Compute input offset for (n,j,h,w) using input strides
            long long in_offset = n * input_n_stride + j * input_c_stride + h * input_h_stride + w * input_w_stride;
            float val = input[in_offset];
            sum_sq += val * val;
        }

        // Compute normalization value
        float denom = k + alpha * sum_sq;
        float inv_denom = powf(denom, -beta); // denom^{-beta}

        // Compute input value at current position
        long long in_offset = n * input_n_stride + c * input_c_stride + h * input_h_stride + w * input_w_stride;
        float x_val = input[in_offset];

        // Write output using output strides
        long long out_offset = n * output_n_stride + c * output_c_stride + h * output_h_stride + w * output_w_stride;
        output[out_offset] = x_val * inv_denom;
    }
}

// Launcher function (C linkage for binding)
extern "C" void cross_map_lrn_forward_launcher(
    const float* input_ptr,
    float* output_ptr,
    int N, int C, int H, int W,
    long long input_n_stride, long long input_c_stride, long long input_h_stride, long long input_w_stride,
    long long output_n_stride, long long output_c_stride, long long output_h_stride, long long output_w_stride,
    int size,
    double alpha, double beta, double k,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit, kernel uses grid-stride loop

    cross_map_lrn_kernel<<<blocks, threads, 0, stream>>>(
        input_ptr, output_ptr,
        N, C, H, W,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        output_n_stride, output_c_stride, output_h_stride, output_w_stride,
        size,
        (float)alpha, (float)beta, (float)k);
}
