// kernels/custom_ops.cu
// Contains non-convolution CUDA kernels and launchers.
// Convolution/transposed convolution are implemented elsewhere using cuDNN.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Kernel: adaptive average pooling over last two dimensions (height, width) to (1,1)
// Input: 5D tensor [batch, channels, depth, height, width], float32
// Output: 5D tensor [batch, channels, depth, 1, 1], float32
// Strides: input may have arbitrary strides, but innermost dimension is contiguous
//          (stride_w == 1). For each (b,c,d), average over all h,w elements.
// -----------------------------------------------------------------------------
__global__ void adaptive_avg_pool2d_5d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_d,
    long long input_stride_h,
    long long input_stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * depth;
    if (idx >= total) return;

    int b = idx / (channels * depth);
    int rem = idx % (channels * depth);
    int c = rem / depth;
    int d = rem % depth;

    const float* base = input
        + b * input_stride_b
        + c * input_stride_c
        + d * input_stride_d;

    float sum = 0.0f;
    for (int h = 0; h < height; ++h) {
        const float* row = base + h * input_stride_h;
        for (int w = 0; w < width; ++w) {
            sum += row[w * input_stride_w];
        }
    }
    int pool_size = height * width;
    output[idx] = sum / static_cast<float>(pool_size);
}

// -----------------------------------------------------------------------------
// Launcher for adaptive_avg_pool2d_5d
// -----------------------------------------------------------------------------
extern "C" void adaptive_avg_pool2d_5d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_d,
    long long input_stride_h,
    long long input_stride_w,
    cudaStream_t stream
) {
    int total = batch * channels * depth;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    adaptive_avg_pool2d_5d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch, channels, depth, height, width,
        input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w
    );
}

// -----------------------------------------------------------------------------
// Kernel: LeakyReLU elementwise (float32)
// -----------------------------------------------------------------------------
__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float val = input[idx];
    output[idx] = val > 0.0f ? val : val * negative_slope;
}

// -----------------------------------------------------------------------------
// Launcher for leaky_relu
// -----------------------------------------------------------------------------
extern "C" void leaky_relu_launcher(
    const float* input,
    float* output,
    int total,
    float negative_slope,
    cudaStream_t stream
) {
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, total, negative_slope
    );
}

// -----------------------------------------------------------------------------
// Kernel: bias add (broadcast bias[c] over depth, height, width)
// Input: 5D tensor [batch, channels, depth, height, width]
// Bias: 4D tensor [channels, 1, 1, 1] (contiguous), but we pass pointer to its data;
//       bias[c] is at bias_ptr[c]
// Since bias shape is [channels,1,1,1], bias[c] == bias_ptr[c].
// We assume bias is contiguous and has same data as a 1D vector of length channels.
// -----------------------------------------------------------------------------
__global__ void bias_add_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    int total,
    int channels,
    int channel_stride_in_elems,  // number of elements per channel (depth*height*width)
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Determine channel index: idx is flattened as (batch, channel, spatial)
    // where spatial = depth*height*width
    int spatial = channel_stride_in_elems;
    int b = idx / (channels * spatial);
    int rem = idx % (channels * spatial);
    int c = rem / spatial;

    output[idx] = input[idx] + bias[c];
}

// -----------------------------------------------------------------------------
// Launcher for bias_add
// -----------------------------------------------------------------------------
extern "C" void bias_add_launcher(
    const float* input,
    float* output,
    const float* bias,  // pointer to bias tensor of shape [channels,1,1,1], contiguous
    int total,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream
) {
    int spatial = depth * height * width;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    bias_add_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, bias, total, channels, spatial, 0 /*unused batch*/
    );
}

// -----------------------------------------------------------------------------
// Kernel: clamp elementwise to [min_val, max_val]
// -----------------------------------------------------------------------------
__global__ void clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    float min_val,
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float val = input[idx];
    val = val < min_val ? min_val : val;
    val = val > max_val ? max_val : val;
    output[idx] = val;
}

// -----------------------------------------------------------------------------
// Launcher for clamp
// -----------------------------------------------------------------------------
extern "C" void clamp_launcher(
    const float* input,
    float* output,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    clamp_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, total, min_val, max_val
    );
}

// Note: No convolution kernels are included; conv3d and conv_transpose3d are
// implemented in separate C++ files using cuDNN.