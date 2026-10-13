// CUDA kernels and launchers for non-convolution custom ops
// Conv2d is handled by cuDNN in a separate C++ binding (not implemented here)
#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// add_bias_forward: out = x + bias, bias shape [C,1,1] broadcast to [N,C,H,W]
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* out, const float* x, const float* bias,
                                int N, int C, int HW) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total = (int64_t)N * C * HW;
    for (int64_t i = idx; i < total; i += stride) {
        int c = (i / HW) % C;
        out[i] = x[i] + bias[c];
    }
}

extern "C" void add_bias_launcher(float* out, const float* x, const float* bias,
                                  int N, int C, int H, int W, cudaStream_t stream) {
    int HW = H * W;
    int64_t total = (int64_t)N * C * HW;
    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535; // grid-stride loop handles the rest
    add_bias_kernel<<<grid, block, 0, stream>>>(out, x, bias, N, C, HW);
}

// -----------------------------------------------------------------------------
// batch_norm_forward (eval mode):
// out = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// weight and bias may be null; if null, use 1 and 0 respectively.
// -----------------------------------------------------------------------------
__global__ void batch_norm_kernel(float* out, const float* x,
                                  const float* weight, const float* bias,
                                  const float* running_mean, const float* running_var,
                                  float eps, int N, int C, int HW) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total = (int64_t)N * C * HW;
    for (int64_t i = idx; i < total; i += stride) {
        int c = (i / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float invstd = rsqrtf(var + eps);
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;
        out[i] = (x[i] - mean) * invstd * w + b;
    }
}

extern "C" void batch_norm_launcher(float* out, const float* x,
                                    const float* weight, const float* bias,
                                    const float* running_mean, const float* running_var,
                                    float eps, int N, int C, int H, int W,
                                    cudaStream_t stream) {
    int HW = H * W;
    int64_t total = (int64_t)N * C * HW;
    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    batch_norm_kernel<<<grid, block, 0, stream>>>(out, x, weight, bias,
                                                   running_mean, running_var, eps,
                                                   N, C, HW);
}

// -----------------------------------------------------------------------------
// avg_pool2d_forward:
// kernel_size = K, stride = S, padding = P, count_include_pad = true
// output spatial size = floor((H + 2P - K)/S) + 1
// The contract uses K=3, S=1, P=1, so output size = H (since 62->62).
// For each output element, compute average over window including padded zeros.
// -----------------------------------------------------------------------------
__global__ void avg_pool2d_kernel(float* out, const float* x,
                                  int N, int C, int H, int W,
                                  int K, int S, int P,
                                  int outH, int outW) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total = (int64_t)N * C * outH * outW;
    for (int64_t i = idx; i < total; i += stride) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int c = (i / (outW * outH)) % C;
        int n = i / (outW * outH * C);
        // input region start
        int h_start = oh * S - P;
        int w_start = ow * S - P;
        float sum = 0.0f;
        for (int kh = 0; kh < K; kh++) {
            int ih = h_start + kh;
            if (ih < 0 || ih >= H) continue; // count_include_pad=True means we still divide by K*K, so skip adding but denominator remains K*K
            for (int kw = 0; kw < K; kw++) {
                int iw = w_start + kw;
                if (iw < 0 || iw >= W) continue;
                int input_idx = ((n * C + c) * H + ih) * W + iw;
                sum += x[input_idx];
            }
        }
        out[i] = sum / (float)(K * K);
    }
}

extern "C" void avg_pool2d_launcher(float* out, const float* x,
                                    int N, int C, int H, int W,
                                    int kernel_size, int stride, int padding,
                                    cudaStream_t stream) {
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;
    int64_t total = (int64_t)N * C * outH * outW;
    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    avg_pool2d_kernel<<<grid, block, 0, stream>>>(out, x, N, C, H, W,
                                                  kernel_size, stride, padding,
                                                  outH, outW);
}

// -----------------------------------------------------------------------------
// sigmoid_forward: out = 1 / (1 + exp(-x))
// -----------------------------------------------------------------------------
__global__ void sigmoid_kernel(float* out, const float* x, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

extern "C" void sigmoid_launcher(float* out, const float* x, int64_t total,
                                 cudaStream_t stream) {
    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    sigmoid_kernel<<<grid, block, 0, stream>>>(out, x, total);
}
