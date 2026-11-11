#include <cuda_runtime.h>

// AvgPool2d kernel: 3x3 window, stride=2, padding=1, count_include_pad=True
__global__ void avg_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N, int C, int H, int W,
                                  int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    int h_start = h_out * 2 - 1;  // stride=2, padding=1
    int w_start = w_out * 2 - 1;

    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < 3; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W) continue;
            // input is contiguous NCHW
            sum += input[((n * C + c) * H + h_in) * W + w_in];
            ++count;
        }
    }
    // count_include_pad=True: divide by kernel_size * kernel_size = 9
    output[idx] = sum / 9.0f;
}

// LogSoftmax along dim=1 (channels) for NCHW tensor
__global__ void log_softmax_dim1_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    int total = N * HW;
    if (idx >= total) return;

    int n = idx / HW;
    int spatial = idx % HW;

    // find max over channel dimension for numerical stability
    float max_val = input[(n * C) * HW + spatial];
    for (int c = 1; c < C; ++c) {
        float val = input[(n * C + c) * HW + spatial];
        if (val > max_val) max_val = val;
    }

    // compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum_exp += expf(input[(n * C + c) * HW + spatial] - max_val);
    }

    // log_sum = max + log(sum_exp)
    float log_sum = max_val + logf(sum_exp);

    // compute log softmax each channel
    for (int c = 0; c < C; ++c) {
        int out_idx = (n * C + c) * HW + spatial;
        output[out_idx] = input[out_idx] - log_sum;
    }
}

// Launcher for avg_pool2d, assumes contiguous NCHW input
// The binding layer should ensure input is contiguous CUDA float32.
extern "C" void avg_pool2d_launcher(const float* input,
                                    float* output,
                                    int N, int C, int H, int W,
                                    int kernel_size, int stride, int padding,
                                    cudaStream_t stream) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Currently only supports kernel_size=3, stride=2, padding=1 (as per contract)
    // Could be generalized if needed, but kept simple for performance.
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out);
}

// Launcher for log_softmax_dim1, assumes contiguous NCHW input
extern "C" void log_softmax_dim1_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           cudaStream_t stream) {
    int HW = H * W;
    int total = N * HW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    log_softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}
