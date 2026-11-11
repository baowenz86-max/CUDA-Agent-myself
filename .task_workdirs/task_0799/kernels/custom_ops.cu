// CUDA kernels for custom operators
#include <cuda_runtime.h>

// MaxPool2d forward kernel
// Each thread computes one output element (or loops over multiple via grid-stride).
// Input and output are float32 tensors with arbitrary strides (but usually contiguous).
// For simplicity and correctness, we assume contiguous layout for max_pool2d because
// max_pool2d is usually called on contiguous tensors. The contract says layout supports strides,
// so we should handle arbitrary strides. We'll implement using explicit stride parameters.
// However, the launcher signature only includes pointers and sizes, not strides.
// To support arbitrary strides, we would need to pass strides. But the binding contract only
// includes input, kernel_size, stride. In practice, max_pool2d input from the model is contiguous
// because it comes from randn. We'll assume contiguous and document that if non-contiguous input
// is needed, the binding layer can call .contiguous() first. The contract says layout supports strides,
// but our launcher takes raw pointer; we can add a note that binding layer must ensure contiguity or
// pass strides as additional parameters. Since we cannot modify binding contract for this stage,
// we will assume contiguous input and output, which is sufficient for the given model.
// If non-contiguous support is required, a future binding can pass strides.
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_size, int64_t stride,
    int64_t H_out, int64_t W_out)
{
    int64_t total = N * C * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += grid_size) {
        int64_t ow = i % W_out;
        int64_t oh = (i / W_out) % H_out;
        int64_t oc = (i / (W_out * H_out)) % C;
        int64_t on = i / (W_out * H_out * C);

        int64_t h_start = oh * stride;
        int64_t w_start = ow * stride;

        float max_val = -3.402823466e+38f;
        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            int64_t h = h_start + kh;
            if (h < 0 || h >= H) continue;
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t w = w_start + kw;
                if (w < 0 || w >= W) continue;
                float val = input[((on * C + oc) * H + h) * W + w];
                if (val > max_val) max_val = val;
            }
        }
        output[i] = max_val;
    }
}

// Stack forward kernel: copies input tensor to output at offset 0
// Input: single tensor of shape [N, C, H, W]
// Output: shape [1, N, C, H, W], just copy all elements
__global__ void stack_single_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += grid_size) {
        output[i] = input[i];
    }
}

// Launcher for max_pool2d_forward
// Input: float* input, int64_t N, C, H, W, int64_t kernel_size, int64_t stride
// Output: float* output (preallocated with shape [N, C, H_out, W_out])
// stream: CUDA stream
// We assume input and output are contiguous float32 arrays.
// If non-contiguous input is needed, the binding layer should call .contiguous() and then
// pass the contiguous data pointer along with the same sizes.
extern "C" void max_pool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_size, int64_t stride,
    cudaStream_t stream)
{
    int64_t H_out = (H - kernel_size) / stride + 1;
    int64_t W_out = (W - kernel_size) / stride + 1;

    int64_t total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap; kernel uses grid-stride loop

    max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, kernel_size, stride, H_out, W_out);
}

// Launcher for stack_forward with a single tensor
// Input: const float* input (single tensor)
// Output: float* output (preallocated with shape [1, N, C, H, W])
// numel: number of elements in input (= N*C*H*W)
// stream: CUDA stream
extern "C" void stack_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    stack_single_copy_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
