#include <cuda_runtime.h>

// SELU activation: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
// alpha = 1.6732632423543772, scale = 1.0507009873554805
__global__ void selu_kernel(float* out, const float* in, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const float alpha = 1.6732632423543772f;
    const float scale = 1.0507009873554805f;

    for (int64_t i = idx; i < numel; i += stride) {
        float x = in[i];
        float val = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        out[i] = scale * val;
    }
}

extern "C" void selu_forward_launcher(
    float* out,
    const float* in,
    int64_t numel,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t max_grid_x = (numel + block_size - 1) / block_size;
    if (max_grid_x < 1) max_grid_x = 1;
    if (max_grid_x > 2147483647LL) max_grid_x = 2147483647LL;
    int grid_x = (int)max_grid_x;
    selu_kernel<<<grid_x, block_size, 0, stream>>>(out, in, numel);
}
