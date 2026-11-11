#include <cuda_runtime.h>

// Helper to get global linear index from 4D tensor with strides
__device__ __forceinline__ int64_t get_linear_index(
    int64_t b, int64_t c, int64_t h, int64_t w,
    int64_t batch_stride, int64_t channel_stride, int64_t height_stride, int64_t width_stride) {
    return b * batch_stride + c * channel_stride + h * height_stride + w * width_stride;
}

// Kernel for nn.Unfold operation with stride=1, padding=0, dilation=1
// Input shape: [batch, channels, height, width]
// Output shape: [batch, channels*kernel_h*kernel_w, output_height*output_width]
// Output is contiguous.
__global__ void unfold_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int out_h,
    int out_w,
    int64_t batch_stride,
    int64_t channel_stride,
    int64_t height_stride,
    int64_t width_stride) {

    int64_t out_numel = (int64_t)batch_size * channels * kernel_h * kernel_w * out_h * out_w;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < out_numel; i += stride) {
        // Decompose i into (b, c, kh, kw, oh, ow)
        int ow = i % out_w;
        int64_t tmp = i / out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int kw = tmp % kernel_w;
        tmp /= kernel_w;
        int kh = tmp % kernel_h;
        tmp /= kernel_h;
        int c = tmp % channels;
        int b = tmp / channels;

        int h_in = oh + kh;   // since dilation=1, padding=0, stride=1
        int w_in = ow + kw;

        int64_t input_index = get_linear_index(b, c, h_in, w_in,
                                               batch_stride, channel_stride, height_stride, width_stride);
        int64_t output_index = i;
        output[output_index] = input[input_index];
    }
}

// Kernel for sum along a dimension.
// This is a naive implementation where each output element is computed by one thread
// summing over the reduction dimension. Optimized for simplicity and correctness.
// Input is assumed to be a 3D tensor (batch, channels*kernel_h*kernel_w, L) for the specific use case,
// but the launcher can handle any tensor shape by treating it as 3D with outer, reduction, inner sizes.
// For general dim sum, we map the tensor to (outer, reduction, inner) where reduction is the target dim.
// The kernel below sums over contiguous reduction dimension when input is contiguous; for arbitrary strides,
// the launcher may first make a contiguous copy. Here we assume input is contiguous (which is true for custom_unfold output).
__global__ void sum_dim_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size) {

    int64_t out_numel = outer_size * inner_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < out_numel; i += stride) {
        int64_t inner = i % inner_size;
        int64_t outer = i / inner_size;
        // input index: outer * reduce_size * inner_size + r * inner_size + inner
        double sum = 0.0;
        for (int64_t r = 0; r < reduce_size; ++r) {
            sum += (double)input[(outer * reduce_size + r) * inner_size + inner];
        }
        output[i] = (float)sum;
    }
}

// Launcher for custom_unfold
// Input tensor x is assumed to be at least 4D, but we only need first 4 dimensions.
// The caller must ensure x is contiguous; if not, the binding will make a contiguous copy before calling.
// Actually contract says supports strides, so we need to handle arbitrary strides. We will pass strides as int64_t values.
// We assume input is float32 CUDA tensor.
extern "C" void custom_unfold_launcher(
    float* output,
    const float* input,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int64_t batch_stride,
    int64_t channel_stride,
    int64_t height_stride,
    int64_t width_stride,
    cudaStream_t stream) {

    int out_h = height - kernel_h + 1;
    int out_w = width - kernel_w + 1;
    int64_t out_numel = (int64_t)batch_size * channels * kernel_h * kernel_w * out_h * out_w;

    // If out_numel is zero, return (empty tensor)
    if (out_numel <= 0) return;

    int threads = 256;
    int blocks = (int)((out_numel + threads - 1) / threads);
    // Cap blocks to a reasonable number to avoid grid dimension overflow; grid-stride loop handles rest.
    if (blocks > 65535) blocks = 65535;  // max grid dim x is 2^31-1 for recent, but 65535 is safe for all.

    unfold_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        batch_size, channels, height, width,
        kernel_h, kernel_w, out_h, out_w,
        batch_stride, channel_stride, height_stride, width_stride);
}

// Launcher for custom_sum_dim
// Assumes input is contiguous (as from custom_unfold) and is 3D (batch, reduce_dim_size, inner)
// We compute sum along reduction dimension.
// For general dim, the binding will permute/reshape to make reduction dimension the middle one and ensure contiguity.
// Actually the contract says supports arbitrary dim, but we can restrict to the use case where input is 3D and dim=1.
// To be safe, we provide a generic launcher that works for any dim by flattening to (outer, reduce, inner).
// We require the input to be contiguous and the reduction dimension to be the one specified by the caller; the binding will handle strides by making contiguous and moving axis.
extern "C" void custom_sum_dim_launcher(
    float* output,
    const float* input,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size,
    cudaStream_t stream) {

    int64_t out_numel = outer_size * inner_size;
    if (out_numel <= 0) return;

    int threads = 256;
    int blocks = (int)((out_numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    sum_dim_kernel<<<blocks, threads, 0, stream>>>(
        output, input, outer_size, reduce_size, inner_size);
}
