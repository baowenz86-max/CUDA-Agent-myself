// custom_ops.cu - CUDA kernels and launchers for max_dim_reduce and sync_bn_inference
#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// max_dim_reduce: reduces over dimension 1 (channels) with keepdim=True.
// Input: x (float32), shape [N, C, H, W] contiguous
// Output: values (float32) shape [N, 1, H, W], indices (int64) shape [N, 1, H, W]
// dim is expected to be 1 (hardcoded due to contract), keepdim true.
// -----------------------------------------------------------------------------

__global__ void max_reduce_kernel(
    const float* __restrict__ x,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int N, int C, int H, int W)
{
    int total_spatial = N * H * W;  // number of output elements (N*1*H*W)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total_spatial; pos += stride) {
        // Decode pos into n, h, w
        int n = pos / (H * W);
        int rem = pos % (H * W);
        int h = rem / W;
        int w = rem % W;

        // Input base pointer for this n: x + n*C*H*W
        const float* base = x + (size_t)n * C * H * W;
        int c_offset = h * W + w;

        float max_val = base[c_offset];
        int max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = base[c * H * W + c_offset];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        values[pos] = max_val;
        indices[pos] = (int64_t)max_idx;
    }
}

extern "C" void max_dim_reduce_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int64_t dim,
    bool keepdim,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    // We only support dim=1 and keepdim=true per contract; other cases are not used.
    // For safety, if dim != 1 or keepdim != true, we do nothing (should not happen).
    if (dim != 1 || !keepdim || N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return;
    }

    int total_spatial = N * H * W;
    int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size? But total_spatial = 512*64*64 = 2,097,152, blocks = 8192.
    // That's fine (max grid dimension 2^31-1).
    max_reduce_kernel<<<blocks, threads, 0, stream>>>(
        x, values, indices, N, C, H, W);
}

// -----------------------------------------------------------------------------
// sync_bn_inference: applies BatchNorm inference using running statistics.
// Input: x (float32), shape [N, C, H, W] contiguous
// weight, bias, running_mean, running_var: float32 arrays of size C
// eps: double
// Output: y (float32) same shape as x
// -----------------------------------------------------------------------------

__global__ void bn_inference_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ y,
    int N, int C, int H, int W,
    double eps)
{
    int total_elements = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // Determine channel index
        int c = (i / (H * W)) % C;
        float scale = weight[c] / sqrtf((float)(running_var[c] + eps));
        float shift = bias[c] - running_mean[c] * scale;
        y[i] = x[i] * scale + shift;
    }
}

extern "C" void sync_bn_inference_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return;
    }

    int total_elements = N * C * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // total_elements = 512*32*64*64 = 67,108,864 -> blocks = 262,144
    // That's fine for modern GPUs.
    bn_inference_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, running_mean, running_var, y, N, C, H, W, eps);
}
