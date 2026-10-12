#include <cuda_runtime.h>
#include <cmath>

// ReflectionPad2d kernel: 输入四维 (N, C, H_in, W_in)，输出 (N, C, H_out, W_out)
// 每个线程处理输出张量中的一个元素
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pad) {
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;  // 每个 z block 对应一个 (n,c) 平面

    if (nc >= N * C || out_h >= H_out || out_w >= W_out) {
        return;
    }

    int n = nc / C;
    int c = nc % C;

    // 反射索引映射
    int in_h = out_h - pad;
    if (in_h < 0) {
        in_h = -in_h;
    } else if (in_h >= H_in) {
        in_h = 2 * (H_in - 1) - in_h;
    }
    // 由于 pad <= H_in-1，上述映射后 in_h 必然在 [0, H_in-1] 内，但为防止极端情况再次修正
    if (in_h < 0) in_h = -in_h;
    if (in_h >= H_in) in_h = 2 * (H_in - 1) - in_h;

    int in_w = out_w - pad;
    if (in_w < 0) {
        in_w = -in_w;
    } else if (in_w >= W_in) {
        in_w = 2 * (W_in - 1) - in_w;
    }
    if (in_w < 0) in_w = -in_w;
    if (in_w >= W_in) in_w = 2 * (W_in - 1) - in_w;

    size_t input_idx = ((size_t)n * C + c) * H_in * W_in + (size_t)in_h * W_in + in_w;
    size_t output_idx = ((size_t)nc) * H_out * W_out + (size_t)out_h * W_out + out_w;
    output[output_idx] = input[input_idx];
}

// erf kernel: y = erf(x)
__global__ void erf_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        output[i] = erff(input[i]);
    }
}

// sin kernel: y = sin(x)
__global__ void sin_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        output[i] = sinf(input[i]);
    }
}

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pad,
    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C);
    reflection_pad2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, pad);
}

extern "C" void erf_launcher(const float* input, float* output, size_t n, cudaStream_t stream) {
    int block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        grid_size = 2147483647;
    }
    erf_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, n);
}

extern "C" void sin_launcher(const float* input, float* output, size_t n, cudaStream_t stream) {
    int block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        grid_size = 2147483647;
    }
    sin_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, n);
}
