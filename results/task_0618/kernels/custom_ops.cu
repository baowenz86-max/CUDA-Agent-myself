#include <cuda_runtime.h>

// 2x2 max pooling with stride 2, no padding.
// input: [N, C, H, W], output: [N, C, H/2, W/2]
__global__ void maxpool2d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int H, int W) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * (H / 2) * (W / 2);
    if (out_idx >= total_out) return;

    int out_w = out_idx % (W / 2);
    int tmp = out_idx / (W / 2);
    int out_h = tmp % (H / 2);
    tmp /= (H / 2);
    int c = tmp % C;
    int n = tmp / C;

    int in_h = out_h * 2;
    int in_w = out_w * 2;
    int base = ((n * C + c) * H + in_h) * W + in_w;

    float max_val = input[base];
    max_val = fmaxf(max_val, input[base + 1]);
    max_val = fmaxf(max_val, input[base + W]);
    max_val = fmaxf(max_val, input[base + W + 1]);

    output[out_idx] = max_val;
}

// Softmax over channel dimension (dim=1).
// Input shape: [N, C, H, W]. For each (n, h, w) we compute softmax over C channels.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int N, int C, int HW) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= N * HW) return;

    int hw = spatial_idx % HW;
    int n = spatial_idx / HW;

    const float* in_base = input + (long long)n * C * HW + hw;
    float* out_base = output + (long long)n * C * HW + hw;

    // Find max for numerical stability
    float max_val = in_base[0];
    for (int c = 1; c < C; ++c) {
        max_val = fmaxf(max_val, in_base[c * HW]);
    }

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float val = __expf(in_base[c * HW] - max_val);
        out_base[c * HW] = val;
        sum += val;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int c = 0; c < C; ++c) {
        out_base[c * HW] *= inv_sum;
    }
}

// Launcher for maxpool2d
// input: float pointer, N*C*H*W elements
// output: float pointer, N*C*(H/2)*(W/2) elements
// H and W must be even.
extern "C" void maxpool2d_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H < 2 || W < 2) return;
    int out_H = H / 2;
    int out_W = W / 2;
    long long total_out = (long long)N * C * out_H * out_W;
    if (total_out == 0) return;

    int threads = 256;
    long long blocks_ll = (total_out + threads - 1) / threads;
    int blocks = (blocks_ll > 2147483647) ? 2147483647 : (int)blocks_ll;

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}

// Launcher for softmax_dim1
// input: float pointer, N*C*H*W elements
// output: float pointer, same size
extern "C" void softmax_dim1_forward_launcher(const float* input,
                                              float* output,
                                              int N, int C, int H, int W,
                                              cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    long long HW = (long long)H * W;
    long long total_spatial = (long long)N * HW;
    if (total_spatial == 0) return;

    int threads = 256;
    long long blocks_ll = (total_spatial + threads - 1) / threads;
    int blocks = (blocks_ll > 2147483647) ? 2147483647 : (int)blocks_ll;

    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, (int)HW);
}
