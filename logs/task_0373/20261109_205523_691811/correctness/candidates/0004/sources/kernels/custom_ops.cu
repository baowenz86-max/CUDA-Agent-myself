#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: compute max over channel dimension (dim=1)
// input: [B, C, H, W], output values: [B, H, W] float, indices: [B, H, W] int64
// Each thread handles one (b, h, w) position and loops over C.
__global__ void max_dim1_kernel(
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    const float* __restrict__ input,
    int B, int C, int H, int W) {
    int total = B * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int b = idx / (W * H);

    const float* base = input + ((long long)b * C * H + (long long)h) * W + w;
    float max_val = -3.402823466e+38F; // -FLT_MAX
    int64_t max_idx = 0;

    for (int c = 0; c < C; ++c) {
        float val = base[(long long)c * H * W];
        if (val > max_val) {
            max_val = val;
            max_idx = c;
        }
    }

    values[idx] = max_val;
    indices[idx] = max_idx;
}

// Kernel: stack input twice along new dim=1
// output shape: [B, 2, C, H, W], each block copies a contiguous chunk.
// We use a 1D grid over total output elements.
__global__ void stack_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int B, int C, int H, int W) {
    long long total = (long long)B * 2LL * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    long long tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    tmp /= C;
    int s = tmp % 2; // which copy (0 or 1)
    int b = tmp / 2;

    // input index: same b, c, h, w
    long long input_idx = ((long long)b * C * H + (long long)c * H + h) * W + w;
    output[idx] = input[input_idx];
}

// Host launchers (extern "C")
extern "C" void custom_max_dim1_launcher(
    float* values,
    int64_t* indices,
    const float* input,
    int B, int C, int H, int W,
    cudaStream_t stream) {
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    max_dim1_kernel<<<blocks, threads, 0, stream>>>(
        values, indices, input, B, C, H, W);
}

extern "C" void custom_stack_dim1_launcher(
    float* output,
    const float* input,
    int B, int C, int H, int W,
    cudaStream_t stream) {
    long long total = (long long)B * 2LL * C * H * W;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    // Grid dimension limited to 2^31-1, but our total may be ~536M, blocks ~2.1M, fine.
    int blocks = (int)blocks_ll;
    stack_dim1_kernel<<<blocks, threads, 0, stream>>>(
        output, input, B, C, H, W);
}
