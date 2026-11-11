#include <cuda_runtime.h>

// cat_forward: concatenate two float32 tensors along dim=1.
// Inputs are contiguous with shapes:
//   x: [batch, Cx, H, W]
//   y: [batch, Cy, H, W]
// Output: [batch, Cx+Cy, H, W]
// dim=1 is assumed (as per contract); other dims are not implemented.
__global__ void cat_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ y,
    int64_t total_elements,
    int64_t Cx,
    int64_t Cy,
    int64_t HW)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        // Determine batch, channel, spatial offset
        int64_t hw = i % HW;
        int64_t rem = i / HW;
        int64_t c = rem % (Cx + Cy);
        int64_t b = rem / (Cx + Cy);

        float val;
        if (c < Cx) {
            // from x
            int64_t x_idx = (b * Cx + c) * HW + hw;
            val = x[x_idx];
        } else {
            // from y
            int64_t c_offset = c - Cx;
            int64_t y_idx = (b * Cy + c_offset) * HW + hw;
            val = y[y_idx];
        }
        out[i] = val;
    }
}

// multiply_scalar_forward: elementwise multiply by scalar
__global__ void multiply_scalar_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t total_elements,
    float scalar)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        out[i] = in[i] * scalar;
    }
}

extern "C" void cat_forward(
    float* output,
    const float* x,
    const float* y,
    int64_t total_elements,
    int64_t Cx,
    int64_t Cy,
    int64_t HW,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap for older GPUs; grid-stride loop handles rest
    cat_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, x, y, total_elements, Cx, Cy, HW);
}

extern "C" void multiply_scalar_forward(
    float* output,
    const float* input,
    int64_t total_elements,
    double scalar,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    multiply_scalar_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements, (float)scalar);
}
