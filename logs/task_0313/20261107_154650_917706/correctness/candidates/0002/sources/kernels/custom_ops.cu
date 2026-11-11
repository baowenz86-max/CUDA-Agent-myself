#include <cuda_runtime.h>

// Utility to compute channel index in NCHW layout
__device__ __forceinline__ int get_channel_index(int idx, int C, int HW) {
    return (idx / HW) % C;
}

// Kernel for BatchNorm2d eval mode: y = (x - mean) / sqrt(var + eps) * weight + bias
// Input and output are contiguous NCHW tensors of float32.
__global__ void batchnorm2d_eval_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int total_elements,
    int C,
    int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        int c = get_channel_index(i, C, HW);
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float val = (input[i] - mean) * inv_std;
        if (weight != nullptr) {
            val *= weight[c];
        }
        if (bias != nullptr) {
            val += bias[c];
        }
        output[i] = val;
    }
}

// Kernel for channel product with keepdim=True. Each thread computes one output element (B*H*W).
// Input is contiguous NCHW, output is contiguous N1HW (effectively [B,1,H,W]).
__global__ void prod_channel_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int C,
    int total_spatial)  // B * H * W
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_spatial) return;

    // For spatial index idx, input base offset is idx * C (since H*W = spatial_size, and contiguous)
    // Actually for NCHW contiguous, the spatial index spans [0, B*H*W);
    // For a given index, batch = idx / HW, spatial = idx % HW, but since C is the first dimension after batch,
    // we can compute the base offset as idx * C.
    // More generally, for NCHW, offset = ((b * C + c) * H + h) * W + w.
    // For fixed b,h,w, varying c gives a stride of H*W. So base = (b * H + h) * W + w = (b * HW + spatial) = idx.
    // Thus base offset = idx * C.
    const float* base_ptr = input + (size_t)idx * C;
    float prod = 1.0f;
    for (int c = 0; c < C; ++c) {
        prod *= base_ptr[c];
    }
    output[idx] = prod;
}

// Host launcher for batchnorm2d_eval_forward
// Parameters:
//   output: pointer to output tensor (allocated by caller)
//   input: pointer to input tensor
//   running_mean, running_var: pointers to buffers of size C
//   weight, bias: pointers to buffers of size C (can be nullptr if not affine)
//   eps: epsilon value
//   B, C, H, W: dimensions
//   stream: CUDA stream
void batchnorm2d_eval_forward_launcher(
    float* output,
    const float* input,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    float eps,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream)
{
    int HW = H * W;
    int total_elements = B * C * HW;
    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to avoid too many blocks; grid-stride loop handles large sizes.
    if (blocks > 65535) blocks = 65535;
    batchnorm2d_eval_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, running_mean, running_var, weight, bias,
        eps, total_elements, C, HW);
}

// Host launcher for prod_channel_forward
// Parameters:
//   output: pointer to output tensor of shape [B,1,H,W] (allocated by caller)
//   input: pointer to input tensor of shape [B,C,H,W]
//   C: number of channels
//   B, H, W: dimensions
//   stream: CUDA stream
void prod_channel_forward_launcher(
    float* output,
    const float* input,
    int B,
    int C,
    int H,
    int W,
    cudaStream_t stream)
{
    int total_spatial = B * H * W;
    const int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    prod_channel_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, C, total_spatial);
}

// Note: The binding layer will call these launchers with appropriate tensor data pointers.
// We use extern "C" to avoid name mangling for easier linking, but C++ linkage is also fine
// as long as declarations match. We'll use extern "C" for clarity.
extern "C" {
    void batchnorm2d_eval_forward(
        float* output,
        const float* input,
        const float* running_mean,
        const float* running_var,
        const float* weight,
        const float* bias,
        float eps,
        int B,
        int C,
        int H,
        int W,
        cudaStream_t stream)
    {
        batchnorm2d_eval_forward_launcher(output, input, running_mean, running_var, weight, bias, eps, B, C, H, W, stream);
    }

    void prod_channel_forward(
        float* output,
        const float* input,
        int B,
        int C,
        int H,
        int W,
        cudaStream_t stream)
    {
        prod_channel_forward_launcher(output, input, B, C, H, W, stream);
    }
}
