#include <cuda_runtime.h>

// Kernel: sum over channel dimension (dim=1) with keepdim=True.
// Input x shape: [batch, C, H, W], output shape: [batch, 1, H, W].
// Assumes contiguous input/output with standard strides.
// Grid: (num_blocks_x, batch), block threads 256, grid-stride loop over hw.
__global__ void sum_channel_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int C,
    int H,
    int W)
{
    long long hw = (long long)H * W;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = idx; i < hw; i += stride) {
        int h = (int)(i / W);
        int w = (int)(i % W);
        int b = blockIdx.y;

        const float* in_base = input + ((long long)b * C + 0) * hw + (long long)h * W + w;
        float sum = 0.0f;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            sum += in_base[c * hw];
        }

        float* out_ptr = output + (long long)b * hw + (long long)h * W + w;
        *out_ptr = sum;
    }
}

// Kernel: elementwise clip to [0,1], shape [batch, 1, H, W].
// Assumes contiguous input/output, so linear index suffices.
__global__ void clip_01_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long total_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = idx; i < total_elements; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        output[i] = val;
    }
}

// Launcher for sum_channel_dim
// Signature: void sum_channel_dim_launcher(const float* x, float* out, int64_t batch, int64_t C, int64_t H, int64_t W, cudaStream_t stream)
extern "C" void sum_channel_dim_launcher(
    const float* x,
    float* out,
    int64_t batch,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream)
{
    // Assuming contiguous input and output.
    int threads = 256;
    long long hw = (long long)H * W;
    int blocks_x = (int)((hw + threads - 1) / threads);
    if (blocks_x > 65535) blocks_x = 65535;
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, (unsigned int)batch);

    sum_channel_dim_kernel<<<grid, threads, 0, stream>>>(
        x, out,
        (int)batch, (int)C, (int)H, (int)W);

    // Kernel launch error check done by binding after call.
}

// Launcher for clip_01
// Signature: void clip_01_launcher(const float* x, float* out, int64_t total_elements, cudaStream_t stream)
extern "C" void clip_01_launcher(
    const float* x,
    float* out,
    int64_t total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    clip_01_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        x, out, total_elements);
}
