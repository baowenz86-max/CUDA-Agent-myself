#include <cuda_runtime.h>
#include <float.h>

// LogSoftmax over channel dimension (dim=1) for contiguous input.
// Input shape: [N,C,D,H,W], output same shape.
// Each thread handles one spatial position across all channels.
__global__ void logsoftmax_channel_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int N, int C, int D, int H, int W) {
    int total_spatial = N * D * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total_spatial; pos += stride) {
        // Calculate n, d, h, w from pos (spatial linear index)
        int w = pos % W;
        int tmp = pos / W;
        int h = tmp % H;
        tmp /= H;
        int d = tmp % D;
        int n = tmp / D;

        // Input base index for this n,d,h,w with channel stride C*D*H*W
        const float* in_base = input + (n * C * D * H * W) + (d * H * W) + (h * W) + w;
        float* out_base = output + (n * C * D * H * W) + (d * H * W) + (h * W) + w;

        float max_val = -FLT_MAX;
        // First pass: find max over channels
        for (int c = 0; c < C; ++c) {
            float val = in_base[c * D * H * W];
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        // Second pass: sum exp(x - max)
        for (int c = 0; c < C; ++c) {
            float val = in_base[c * D * H * W];
            sum_exp += __expf(val - max_val);
        }
        float log_sum = __logf(sum_exp);

        // Write log_softmax values
        for (int c = 0; c < C; ++c) {
            float val = in_base[c * D * H * W];
            out_base[c * D * H * W] = val - max_val - log_sum;
        }
    }
}

// 3D Max Pooling kernel.
// Input shape: [N,C,D,H,W], output shape computed using floor formula.
// Each thread computes one output element.
__global__ void maxpool3d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int D, int H, int W,
                                 int D_out, int H_out, int W_out,
                                 int kernel_d, int kernel_h, int kernel_w,
                                 int stride_d, int stride_h, int stride_w,
                                 int padding_d, int padding_h, int padding_w) {
    int total_out = N * C * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total_out; pos += stride) {
        int w_out = pos % W_out;
        int tmp = pos / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int d_out = tmp % D_out;
        tmp /= D_out;
        int c = tmp % C;
        int n = tmp / C;

        // Compute input window start
        int d_start = d_out * stride_d - padding_d;
        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;

        float max_val = -FLT_MAX;
        bool has_valid = false;

        for (int kd = 0; kd < kernel_d; ++kd) {
            int d_in = d_start + kd;
            if (d_in < 0 || d_in >= D) continue;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h_in = h_start + kh;
                if (h_in < 0 || h_in >= H) continue;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_start + kw;
                    if (w_in < 0 || w_in >= W) continue;
                    // Input index for valid positions
                    int in_idx = (((n * C + c) * D + d_in) * H + h_in) * W + w_in;
                    float val = input[in_idx];
                    if (val > max_val) max_val = val;
                    has_valid = true;
                }
            }
        }

        if (has_valid) {
            int out_idx = (((n * C + c) * D_out + d_out) * H_out + h_out) * W_out + w_out;
            output[out_idx] = max_val;
        } else {
            // Should not happen with valid kernel/stride/padding, but leave as -inf
            int out_idx = (((n * C + c) * D_out + d_out) * H_out + h_out) * W_out + w_out;
            output[out_idx] = -FLT_MAX;
        }
    }
}

// Launcher for logsoftmax_forward: input/output raw pointers, dimensions, stream
extern "C" void logsoftmax_forward_launcher(const float* input,
                                            float* output,
                                            int N, int C, int D, int H, int W,
                                            cudaStream_t stream) {
    int total_spatial = N * D * H * W;
    int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles larger
    logsoftmax_channel_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, D, H, W);
}

// Launcher for maxpool3d_forward
extern "C" void maxpool3d_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int D, int H, int W,
                                           int kernel_d, int kernel_h, int kernel_w,
                                           int stride_d, int stride_h, int stride_w,
                                           int padding_d, int padding_h, int padding_w,
                                           cudaStream_t stream) {
    // Compute output dimensions using floor formula
    int D_out = (D + 2 * padding_d - kernel_d) / stride_d + 1;
    int H_out = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * padding_w - kernel_w) / stride_w + 1;

    int total_out = N * C * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    maxpool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, D, H, W,
        D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w
    );
}
