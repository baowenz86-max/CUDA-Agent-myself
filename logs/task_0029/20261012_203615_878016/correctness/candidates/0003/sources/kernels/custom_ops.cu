#include <cuda_runtime.h>

__global__ void fused_unbind_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int B, int C, int H, int W)
{
    int w = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = tid; pos < B * H; pos += stride) {
        int b = pos / H;
        int h = pos - b * H;

        float max_val = -3.402823466e+38f;
        int64_t max_idx = 0;
        for (int c = 0; c < C; ++c) {
            float val = input[((long long)(b * C + c) * H + h) * W + w];
            if (c == 0 || val > max_val) {
                max_val = val;
                max_idx = (int64_t)c;
            }
        }

        long long out_idx = ((long long)(b * H) + h) * W + w;
        values[out_idx] = max_val;
        indices[out_idx] = max_idx;
    }
}

extern "C" void custom_unbind_max_launcher(
    const float* input,
    float* values,
    int64_t* indices,
    int B, int C, int H, int W,
    cudaStream_t stream)
{
    dim3 block(256);
    int threads_per_output = 256;
    int grid_x = (B * H + threads_per_output - 1) / threads_per_output;
    dim3 grid(grid_x, W);

    fused_unbind_max_kernel<<<grid, block, 0, stream>>>(
        input, values, indices, B, C, H, W);
}
