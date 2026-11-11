#include <cuda_runtime.h>

__global__ void batch_norm_eval_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N,
    int C,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (W * H)) % C;
        int n = i / (W * H * C);

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);

        float x_val = input[i];
        float y = (x_val - mean) * inv_std;

        if (weight != nullptr) {
            y *= weight[c];
        }
        if (bias != nullptr) {
            y += bias[c];
        }

        output[i] = y;
    }
}

extern "C" void batch_norm_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream
) {
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;
    }
    batch_norm_eval_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var,
        eps, N, C, H, W
    );
}
