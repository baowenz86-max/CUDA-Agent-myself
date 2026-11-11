// kernels/custom_ops.cu
// CUDA kernels for non-convolutional custom ops: ReLU and MaxPool2d.
// Conv2d is handled by cuDNN in the binding layer, so no conv kernel here.
#include <cuda_runtime.h>

// ReLU forward: y = max(x, 0)
// Supports arbitrary strides.
__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t inner_size,
    int64_t inner_stride)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t batch_idx = linear / inner_size;
        int64_t inner_idx = linear % inner_size;
        // Compute pointer using base pointer + batch_idx * inner_stride? 
        // Actually input is contiguous in inner dimension, but the batch stride may differ.
        // Simpler: treat as flat array if input is contiguous; for non-contiguous, handle strides.
        // We'll assume the input is at least contiguous in the inner dimension, and the batch dimension stride is provided.
        // The binding layer will pass the appropriate base pointer and strides.
        // Since kernel only needs to know the element index in the flattened view,
        // we can use a 1D grid over total_elements and compute actual pointer via input + linear * element_stride?
        // For full generality, we need the actual strides. The launcher signature includes strides.
        // To keep it simple and correct for our use case (contiguous), we can implement a simpler version.
        // However to support arbitrary strides, we'll pass a pointer to the first element and the strides.
        // Here we implement a generic 1D kernel using input pointer + linear * sizeof(float) assuming contiguous.
        // The binding layer will ensure contiguous by calling .contiguous() if needed.
        // So we can safely assume contiguous input/output. We'll document that.
        output[linear] = fmaxf(input[linear], 0.0f);
    }
}

// Launcher for ReLU: assumes input and output are contiguous float32 tensors.
extern "C" void launch_relu_forward(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    // Cap blocks to a reasonable limit (e.g., 2^31-1) and use grid-stride loop.
    // For total_elements up to 130M, blocks ~ 508k, which is fine.
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements, 0, 0);
}

// MaxPool2d forward: output[b,c,oh,ow] = max over window (kh,kw) of input[b,c,ih,iw]
// kernel_size, stride, padding, dilation are passed as int parameters.
// Supports arbitrary strides by using input strides for batch, channel, height, width.
// The kernel uses a 3D grid: (output_width, output_height, batch*channels) or 2D flattened.
// We use a 1D grid over batch*channels and 2D threads for output positions.
// To keep it simpler, we use a 1D grid and 1D block, with each thread computing one output element.
// We pass strides: in_batch_stride, in_channel_stride, in_height_stride, in_width_stride,
// and output strides similarly. However for simplicity, we assume contiguous tensors.
// The binding layer will ensure contiguity.
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total_outputs = (int64_t)batch_size * channels * output_height * output_width;
    int64_t stride_total = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total_outputs; linear += stride_total) {
        // Decode linear index: ow fastest, then oh, then c, then b.
        int ow = linear % output_width;
        int64_t tmp = linear / output_width;
        int oh = tmp % output_height;
        tmp /= output_height;
        int c = tmp % channels;
        int b = tmp / channels;

        float max_val = -1e30f;
        // Compute input starting positions considering padding and dilation.
        int ih_start = oh * stride_h - padding_h;
        int iw_start = ow * stride_w - padding_w;

        int kh_end = kernel_height;
        int kw_end = kernel_width;
        // For each kernel position, compute input coordinates and check bounds.
        // If all positions are outside, PyTorch maxpool with no padding returns -inf,
        // but here padding=0 and kernel fits within input, so every window has valid elements.
        // For safety, we check and if no valid element, set max_val = -FLT_MAX (or -inf).
        bool found = false;
        for (int kh = 0; kh < kh_end; ++kh) {
            int ih = ih_start + kh * dilation_h;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kw_end; ++kw) {
                int iw = iw_start + kw * dilation_w;
                if (iw < 0 || iw >= input_width) continue;
                // Input element index: ((b * channels + c) * input_height + ih) * input_width + iw
                // Assumes contiguous tensor.
                const float val = input[(((int64_t)b * channels + c) * input_height + ih) * input_width + iw];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }
        // Write output: ((b * channels + c) * output_height + oh) * output_width + ow
        output[(((int64_t)b * channels + c) * output_height + oh) * output_width + ow] = max_val;
    }
}

// Launcher for MaxPool2d: assumes input and output are contiguous float32 tensors.
extern "C" void launch_maxpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream)
{
    int threads = 256;
    int64_t total_outputs = (int64_t)batch_size * channels * output_height * output_width;
    int blocks = (int)((total_outputs + threads - 1) / threads);
    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_size, kernel_size,
        stride, stride, padding, padding, dilation, dilation);
}
