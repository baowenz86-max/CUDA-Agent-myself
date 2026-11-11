// kernels/custom_ops.cu
// CUDA kernels and launchers for custom_gelu and custom_mean_dim1.
// All operators expect float32 CUDA tensors with arbitrary strides.
// Launchers are declared extern "C" for the binding layer.

#include <cuda_runtime.h>
#include <cmath>

// -----------------------------------------------------------------------------
// custom_gelu: y = x * 0.5 * (1 + erf(x / sqrt(2)))
// Elementwise, supports non-contiguous input via strides.
// -----------------------------------------------------------------------------
__global__ void custom_gelu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long numel,
    long long stride_batch,
    long long stride_channel,
    long long stride_height,
    long long stride_width)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Decompose linear index into 4D coordinates (batch, channel, height, width)
    long long b = idx / (stride_batch == 0 ? 1 : stride_channel * 64 * 128); // placeholder, not used safely
    // Avoid division by zero and compute coordinates using strides.
    // We assume dimensions are known: batch=512, channel=64, height=64, width=128 as per contract.
    // But to be general, we can compute coords from strides if they are positive.
    // Here we use a simple division approach assuming the strides reflect the shape.
    // For arbitrary strides, we cannot infer dimensions without extra parameters; however the contract fixes shape [512,64,64,128].
    // We'll pass the strides and also assume the shape via linear idx mapping:
    // idx = b*stride_batch + c*stride_channel + h*stride_height + w*stride_width
    // But given only numel, we can still compute using the known fixed dims? Better to pass shape parameters.
    // To keep launcher simple, we pass dims as constants from binding or compute via known shape.
    // Since the contract is fixed for this model, we hardcode dims in launcher? That would be brittle.
    // Actually we can pass dims: B, C, H, W explicitly. Let's add those parameters for clarity.
    // To avoid confusion, we'll rewrite kernel signature to include B,C,H,W.
    // However the current signature lacks dims, so we cannot decompose properly.
    // We need to include dims in the kernel for correct indexing.
    // We'll fix by including B,C,H,W in launcher and kernel.
    // For now, we'll use a simpler approach: since input is always full 4D with strides,
    // we can compute coordinates by successive modulo/div using known shapes.
    // We'll assume the shape is passed as constants: B=512, C=64, H=64, W=128.
    // But that's not general. Better to modify kernel signature to accept dims.
    // Let's do that in the actual code below.
}

// Simpler: pass shape parameters as int, and use them for decomposition.
__global__ void custom_gelu_kernel_v2(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long numel,
    int B, int C, int H, int W,
    long long stride_batch,
    long long stride_channel,
    long long stride_height,
    long long stride_width)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int w = idx % W;
    long long tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int b = tmp / C;

    long long off = b * stride_batch + c * stride_channel + h * stride_height + w * stride_width;
    float val = x[off];
    y[off] = val * 0.5f * (1.0f + erff(val * 0.70710678118654752440f));
}

// -----------------------------------------------------------------------------
// custom_mean_dim1: y[b,h,w] = mean over c of x[b,c,h,w]
// Supports non-contiguous input/output via strides.
// -----------------------------------------------------------------------------
__global__ void custom_mean_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int C, int H, int W,
    long long stride_batch_in, long long stride_channel_in, long long stride_height_in, long long stride_width_in,
    long long stride_batch_out, long long stride_height_out, long long stride_width_out)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long out_numel = (long long)B * H * W;
    if (idx >= out_numel) return;

    int w = idx % W;
    long long tmp = idx / W;
    int h = tmp % H;
    int b = tmp / H;

    float sum = 0.0f;
    long long in_base = b * stride_batch_in + h * stride_height_in + w * stride_width_in;
    #pragma unroll 4
    for (int c = 0; c < C; ++c) {
        sum += x[in_base + c * stride_channel_in];
    }
    long long out_off = b * stride_batch_out + h * stride_height_out + w * stride_width_out;
    y[out_off] = sum / (float)C;
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------
extern "C" void custom_gelu_launcher(
    const float* x,
    float* y,
    long long numel,
    int B, int C, int H, int W,
    long long stride_batch,
    long long stride_channel,
    long long stride_height,
    long long stride_width,
    cudaStream_t stream)
{
    if (numel == 0) return;
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    custom_gelu_kernel_v2<<<(unsigned int)blocks, threads, 0, stream>>>(
        x, y, numel, B, C, H, W,
        stride_batch, stride_channel, stride_height, stride_width);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Binding layer can check; we just ignore or printf. Better to not printf in library.
        // Optionally, we could throw, but no exception support in .cu without torch.
        // So we leave it: binding will check after launch.
    }
}

extern "C" void custom_mean_dim1_launcher(
    const float* x,
    float* y,
    int B, int C, int H, int W,
    long long stride_batch_in, long long stride_channel_in, long long stride_height_in, long long stride_width_in,
    long long stride_batch_out, long long stride_height_out, long long stride_width_out,
    cudaStream_t stream)
{
    long long out_numel = (long long)B * H * W;
    if (out_numel == 0) return;
    int threads = 256;
    long long blocks = (out_numel + threads - 1) / threads;
    custom_mean_dim1_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        x, y, B, C, H, W,
        stride_batch_in, stride_channel_in, stride_height_in, stride_width_in,
        stride_batch_out, stride_height_out, stride_width_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // no-op; binding can check
    }
}
