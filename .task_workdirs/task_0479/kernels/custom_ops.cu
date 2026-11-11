#include <cuda_runtime.h>

// Elementwise clamp to [0,1]
__global__ void clamp_kernel(float* __restrict__ out, const float* __restrict__ in, long long numel) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < numel; i += stride) {
        float v = in[i];
        out[i] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}

// 2x2 max pooling with stride 2, no padding
__global__ void maxpool2x2_kernel(float* __restrict__ out, const float* __restrict__ in,
                                  int N, int C, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * (H / 2) * (W / 2);
    int stride = gridDim.x * blockDim.x;
    int OH = H / 2;
    int OW = W / 2;
    for (; index < total; index += stride) {
        int ow = index % OW;
        int oh = (index / OW) % OH;
        int c = (index / (OW * OH)) % C;
        int n = index / (OW * OH * C);
        int in_h = oh * 2;
        int in_w = ow * 2;
        int base = ((n * C + c) * H + in_h) * W + in_w;
        float max_val = in[base];
        max_val = fmaxf(max_val, in[base + 1]);
        max_val = fmaxf(max_val, in[base + W]);
        max_val = fmaxf(max_val, in[base + W + 1]);
        out[index] = max_val;
    }
}

extern "C" void clamp_forward_launcher(float* out, const float* in, long long numel, cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    int blocks = (int)min((long long)blocks_ll, (long long)65535); // grid limit
    if (blocks == 0) return;
    clamp_kernel<<<blocks, threads, 0, stream>>>(out, in, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, we should report error; here we simply return silently.
        // The binding layer will handle error checking after launch.
    }
}

extern "C" void maxpool2d_forward_launcher(float* out, const float* in,
                                           int N, int C, int H, int W, cudaStream_t stream) {
    int OH = H / 2;
    int OW = W / 2;
    long long total = (long long)N * C * OH * OW;
    if (total == 0) return;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    int blocks = (int)min(blocks_ll, (long long)2147483647); // max grid dim for x
    maxpool2x2_kernel<<<blocks, threads, 0, stream>>>(out, in, N, C, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling done in binding
    }
}
