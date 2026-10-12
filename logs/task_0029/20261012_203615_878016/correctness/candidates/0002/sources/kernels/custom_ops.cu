#include <cuda_runtime.h>

// Kernel: 对输入形状 (B, C, H, W) 执行 unbind(dim=-1) + max(dim=1)
// 输出: values (B, H, W) float32, indices (B, H, W) int64
// 每个 block 处理一个 W 切片 (w 由 blockIdx.y 决定)
// 每个线程处理一个 (b, h) 输出位置，串行比较 C 个元素，记录第一个最大值索引
__global__ void fused_unbind_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    long long* __restrict__ indices,
    int B, int C, int H, int W)
{
    int w = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = tid; pos < B * H; pos += stride) {
        int b = pos / H;
        int h = pos - b * H;

        float max_val = -3.402823466e+38f; // -FLT_MAX
        long long max_idx = 0;
        for (int c = 0; c < C; ++c) {
            // 输入布局: input[b][c][h][w] = input[(b*C + c)*H*W + h*W + w]
            float val = input[((long long)(b * C + c) * H + h) * W + w];
            if (c == 0 || val > max_val) {
                max_val = val;
                max_idx = (long long)c;
            }
        }

        // 输出布局: values[b][h][w] = values[(b*H + h)*W + w]
        long long out_idx = ((long long)(b * H) + h) * W + w;
        values[out_idx] = max_val;
        indices[out_idx] = max_idx;
    }
}

// Launcher: 由 binding cpp 调用
// 参数: input 指针, values 指针, indices 指针, B, C, H, W, stream
extern "C" void custom_unbind_max_launcher(
    const float* input,
    float* values,
    long long* indices,
    int B, int C, int H, int W,
    cudaStream_t stream)
{
    dim3 block(256);
    // grid.x: 覆盖 B*H 个输出的线程
    int threads_per_output = 256;
    int grid_x = (B * H + threads_per_output - 1) / threads_per_output;
    // grid.y: 每个 W 切片一个 block
    dim3 grid(grid_x, W);

    fused_unbind_max_kernel<<<grid, block, 0, stream>>>(
        input, values, indices, B, C, H, W);
}
