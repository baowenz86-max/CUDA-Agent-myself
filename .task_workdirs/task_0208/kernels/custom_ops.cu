#include <cuda_runtime.h>
#include <math.h>

// BatchNorm2d inference forward kernel
// Input x: [N, C, H, W], float32
// Parameters weight, bias, running_mean, running_var: [C], float32
// eps: scalar float
// y[n,c,h,w] = (x[n,c,h,w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
__global__ void batchnorm2d_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W)
{
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int c = (i / (H * W)) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = weight[c];
        float beta = bias[c];
        y[i] = (x[i] - mean) * inv_std * gamma + beta;
    }
}

// LogSoftmax over dim=1 kernel
// Input x: [N, C, H, W], float32
// For each (n,h,w) location, compute:
//   max_c = max_c x[n,c,h,w]
//   log_sum_exp = log(sum_c exp(x[n,c,h,w] - max_c))
//   y[n,c,h,w] = x[n,c,h,w] - max_c - log_sum_exp
// Spatial dimensions H*W locations per batch, total thread elements = N*H*W
__global__ void logsoftmax_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int HW = H * W;
    int total_positions = N * HW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Each thread independently processes one spatial location (n,h,w), loops over channels
    for (int pos = idx; pos < total_positions; pos += stride) {
        int n = pos / HW;
        int hw = pos % HW;
        // base offset for channel 0 at this n,hw
        const float* x_base = x + (size_t)n * (size_t)C * HW + hw;
        float* y_base = y + (size_t)n * (size_t)C * HW + hw;

        // compute channel max
        float max_val = x_base[0];
        for (int c = 1; c < C; ++c) {
            float val = x_base[(size_t)c * HW];
            if (val > max_val) max_val = val;
        }

        // compute log sum exp (stable)
        float sum_exps = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum_exps += expf(x_base[(size_t)c * HW] - max_val);
        }
        float log_sum = logf(sum_exps);

        // write log softmax for all channels
        for (int c = 0; c < C; ++c) {
            y_base[(size_t)c * HW] = x_base[(size_t)c * HW] - max_val - log_sum;
        }
    }
}

// Launcher for batch norm forward
// Signature must match model_new.py call order:
// custom_batchnorm2d_forward(x, weight, bias, running_mean, running_var, eps)
extern "C" void custom_batchnorm2d_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // cap grid to a reasonable number to avoid launch failure; grid-stride loop handles the rest
    if (grid_size > 65535) grid_size = 65535;
    batchnorm2d_inference_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var, eps, N, C, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In production binding, should check error; here we can't throw easily without torch headers.
        // We leave this empty; binding layer may check via cudaGetLastError after launch.
        // But to avoid compiler warnings, we can ignore.
        (void)err;
    }
}

// Launcher for log softmax dim1 forward
// Signature: custom_log_softmax_dim1_forward(x) -> output
extern "C" void custom_log_softmax_dim1_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int HW = H * W;
    int total_positions = N * HW;
    if (total_positions <= 0) return;
    const int block_size = 256;
    int grid_size = (total_positions + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    logsoftmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, N, C, H, W);
    cudaError_t err = cudaGetLastError();
    (void)err;
}
