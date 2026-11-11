// CUDA kernels for sum_dim1_keepdim and maxpool3d
// No PyTorch dependencies, only raw pointers and cudaStream_t
#include <cuda_runtime.h>
#include <cstdio>

// Helper to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        cudaDeviceReset(); \
    } \
} while(0)

// Kernel: sum over dimension 1 (channels) with keepdim=True
// Input: 5D tensor [batch, channels, depth, height, width] with arbitrary strides
// Output: 5D tensor [batch, 1, depth, height, width] contiguous (strides from shape)
// Each thread computes one output element (b,0,d,h,w) by summing over all channels
template <typename scalar_t>
__global__ void sum_dim1_keepdim_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long in_stride_b,   // stride for batch
    long long in_stride_c,   // stride for channel
    long long in_stride_d,   // stride for depth
    long long in_stride_h,   // stride for height
    long long in_stride_w,   // stride for width
    long long out_stride_b,  // output strides (contiguous assumed)
    long long out_stride_d,
    long long out_stride_h,
    long long out_stride_w)
{
    // Total number of output elements = batch * depth * height * width
    int total = batch * depth * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute coordinates from linear index (assuming output layout [batch,1,depth,height,width])
    int w = idx % width;
    int tmp1 = idx / width;
    int h = tmp1 % height;
    tmp1 = tmp1 / height;
    int d = tmp1 % depth;
    int b = tmp1 / depth;

    // Sum over channels
    scalar_t sum = 0;
    const scalar_t* base = input + b * in_stride_b + d * in_stride_d + h * in_stride_h + w * in_stride_w;
    for (int c = 0; c < channels; ++c) {
        sum += base[c * in_stride_c];
    }
    output[b * out_stride_b + d * out_stride_d + h * out_stride_h + w * out_stride_w] = sum;
}

// Kernel: 3D max pooling with given kernel and stride sizes (no padding, dilation=1)
// Input: 5D tensor [batch, channels, depth, height, width] with arbitrary strides
// Output: 5D tensor [batch, channels, out_depth, out_height, out_width] contiguous
// Each thread computes one output element (b,c,od,oh,ow) as max over window
template <typename scalar_t>
__global__ void maxpool3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kd, int kh, int kw,   // kernel sizes
    int sd, int sh, int sw,   // stride sizes
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_d,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_d,
    long long out_stride_h,
    long long out_stride_w)
{
    int total = batch * channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute coordinates
    int ow = idx % out_width;
    int tmp1 = idx / out_width;
    int oh = tmp1 % out_height;
    tmp1 = tmp1 / out_height;
    int od = tmp1 % out_depth;
    tmp1 = tmp1 / out_depth;
    int c = tmp1 % channels;
    int b = tmp1 / channels;

    // Compute input window start (no padding, so start = od*sd etc.)
    int start_d = od * sd;
    int start_h = oh * sh;
    int start_w = ow * sw;

    scalar_t max_val = -3.402823466e+38f; // -FLT_MAX approximation for scalar_t; better use numeric_limits but avoid <limits> include for device? We can use -1e30f for float only.
    // Since only float32 path needed, hardcode negative infinity.
    // Make sure kernel only instantiated for float.
    const scalar_t* base = input + b * in_stride_b + c * in_stride_c;
    for (int kz = 0; kz < kd; ++kz) {
        int d = start_d + kz;
        if (d >= in_depth) continue;
        const scalar_t* base_d = base + d * in_stride_d;
        for (int ky = 0; ky < kh; ++ky) {
            int h = start_h + ky;
            if (h >= in_height) continue;
            const scalar_t* base_h = base_d + h * in_stride_h;
            for (int kx = 0; kx < kw; ++kx) {
                int w = start_w + kx;
                if (w >= in_width) continue;
                scalar_t val = base_h[w * in_stride_w];
                if (val > max_val) max_val = val;
            }
        }
    }

    // If window entirely out of bounds? With no padding and stride within bounds, window always partially in bounds because output indices computed such that last window fits, but some kz may go out; we handle with bounds check. max_val remains -FLT_MAX if no valid element, but that shouldn't happen for valid output sizes. To be safe, we could clamp to zero but that is incorrect. Since output shape computed by reference, every output has at least one valid element. We'll trust that.
    output[b * out_stride_b + c * out_stride_c + od * out_stride_d + oh * out_stride_h + ow * out_stride_w] = max_val;
}

// Launcher for sum_dim1_keepdim
// Input shape: batch, channels, depth, height, width; arbitrary strides
// Output: contiguous tensor with shape [batch,1,depth,height,width], dtype float32
extern "C" void sum_dim1_keepdim_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_d,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_d,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream)
{
    int total = batch * depth * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    sum_dim1_keepdim_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, depth, height, width,
        in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        out_stride_b, out_stride_d, out_stride_h, out_stride_w);
    CUDA_CHECK(cudaGetLastError());
}

// Launcher for maxpool3d
// Input shape: batch, channels, in_depth, in_height, in_width; arbitrary strides
// Output: contiguous tensor with shape [batch, channels, out_depth, out_height, out_width]
// kernel_size and stride are passed as six ints (kd,kh,kw,sd,sh,sw)
extern "C" void maxpool3d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kd, int kh, int kw,
    int sd, int sh, int sw,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_d,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_d,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream)
{
    int total = batch * channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool3d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kd, kh, kw, sd, sh, sw,
        in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w);
    CUDA_CHECK(cudaGetLastError());
}
