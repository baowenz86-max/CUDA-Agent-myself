#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Adaptive average pooling over last two dimensions (height, width)
// Input: x shape [batch, channels, depth, height, width]
// Output: y shape [batch, channels, depth, 1, 1]
// Each output element is the mean of all elements in the corresponding
// (batch, channel, depth) slice over height and width.
// -----------------------------------------------------------------------------
extern "C" __global__ void adaptive_avg_pool2d_5d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long total_output_elements)  // batch * channels * depth
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total_output_elements; i += stride) {
        // i = ((b * channels) + c) * depth + d
        int d = i % depth;
        long long tmp = i / depth;
        int c = tmp % channels;
        int b = tmp / channels;

        long long base = ((long long)b * channels + c) * depth * height * width
                         + (long long)d * height * width;
        float sum = 0.0f;
        for (int hw = 0; hw < height * width; ++hw) {
            sum += x[base + hw];
        }
        y[i] = sum / (float)(height * width);
    }
}

extern "C" void adaptive_avg_pool2d_5d_launcher(
    const float* x,
    float* y,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream)
{
    long long total = (long long)batch * channels * depth;
    if (total <= 0) return;

    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    adaptive_avg_pool2d_5d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, batch, channels, depth, height, width, total);
}

// -----------------------------------------------------------------------------
// LeakyReLU elementwise: y = x >= 0 ? x : x * negative_slope
// Input and output have the same shape and layout (contiguous assumed by caller)
// -----------------------------------------------------------------------------
extern "C" __global__ void leaky_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long n,
    float negative_slope)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < n; i += stride) {
        float val = x[i];
        y[i] = val >= 0.0f ? val : val * negative_slope;
    }
}

extern "C" void leaky_relu_launcher(
    const float* x,
    float* y,
    long long n,
    float negative_slope,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int threads = 256;
    long long blocks_ll = (n + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(x, y, n, negative_slope);
}

// -----------------------------------------------------------------------------
// Add bias with shape [channels, 1, 1, 1] broadcast over [batch, channels, depth, height, width]
// For each element, add bias[channel]
// -----------------------------------------------------------------------------
extern "C" __global__ void add_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    long long total,       // total elements = batch * channels * depth * height * width
    int channels,
    int inner_size)        // depth * height * width
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        // i = (b * channels + c) * inner_size + offset
        int c = (i / inner_size) % channels;
        y[i] = x[i] + bias[c];
    }
}

extern "C" void add_bias_launcher(
    const float* x,
    const float* bias,
    float* y,
    long long total,
    int channels,
    int inner_size,
    cudaStream_t stream)
{
    if (total <= 0) return;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    add_bias_kernel<<<blocks, threads, 0, stream>>>(x, bias, y, total, channels, inner_size);
}

// -----------------------------------------------------------------------------
// Clamp elementwise: y = min(max(x, min_val), max_val)
// -----------------------------------------------------------------------------
extern "C" __global__ void clamp_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long n,
    float min_val,
    float max_val)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < n; i += stride) {
        float val = x[i];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        y[i] = val;
    }
}

extern "C" void clamp_launcher(
    const float* x,
    float* y,
    long long n,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int threads = 256;
    long long blocks_ll = (n + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    clamp_kernel<<<blocks, threads, 0, stream>>>(x, y, n, min_val, max_val);
}

// -----------------------------------------------------------------------------
// 3D Convolution kernel (without cuDNN)
// Input x: [batch, in_channels, depth, height, width]
// Weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
// Output: [batch, out_channels, out_depth, out_height, out_width]
// Assumes stride=1, padding=1, dilation=1 (as in this task)
// -----------------------------------------------------------------------------
extern "C" __global__ void conv3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int batch, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int out_depth, int out_height, int out_width)
{
    int total_output = batch * out_channels * out_depth * out_height * out_width;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_output; i += stride) {
        int w = i % out_width;
        int h = (i / out_width) % out_height;
        int d = (i / (out_width * out_height)) % out_depth;
        int oc = (i / (out_width * out_height * out_depth)) % out_channels;
        int b = i / (out_width * out_height * out_depth * out_channels);

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int in_d = d * stride_d - pad_d + kd;
                        int in_h = h * stride_h - pad_h + kh;
                        int in_w = w * stride_w - pad_w + kw;
                        if (in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                            long long x_idx = (((long long)b * in_channels + ic) * in_depth + in_d) * in_height * in_width + in_h * in_width + in_w;
                            long long w_idx = (((long long)oc * in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                            sum += x[x_idx] * weight[w_idx];
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        y[i] = sum;
    }
}

extern "C" void conv3d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int batch, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int out_depth, int out_height, int out_width,
    cudaStream_t stream)
{
    int total = batch * out_channels * out_depth * out_height * out_width;
    if (total <= 0) return;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    conv3d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y,
        batch, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
        out_depth, out_height, out_width);
}

// -----------------------------------------------------------------------------
// 3D Transposed Convolution kernel (without cuDNN)
// Input x: [batch, in_channels, depth, height, width]
// Weight: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
// Output: [batch, out_channels, out_depth, out_height, out_width]
// with stride, padding, output_padding as given.
// Assumes stride=1, padding=1, output_padding=0, dilation=1 (as in this task)
// -----------------------------------------------------------------------------
extern "C" __global__ void conv_transpose3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int batch, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int out_depth, int out_height, int out_width,
    int output_pad_d, int output_pad_h, int output_pad_w)
{
    int total_output = batch * out_channels * out_depth * out_height * out_width;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_output; i += stride) {
        int w = i % out_width;
        int h = (i / out_width) % out_height;
        int d = (i / (out_width * out_height)) % out_depth;
        int oc = (i / (out_width * out_height * out_depth)) % out_channels;
        int b = i / (out_width * out_height * out_depth * out_channels);

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        // For transposed conv, the effective input position is:
                        // in_pos = (out_pos + pad - kernel_index) / stride
                        // must be integer and within input bounds.
                        int in_d = d + pad_d - kd;
                        int in_h = h + pad_h - kh;
                        int in_w = w + pad_w - kw;
                        if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                            in_d /= stride_d;
                            in_h /= stride_h;
                            in_w /= stride_w;
                            if (in_d >= 0 && in_d < in_depth && in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                long long x_idx = (((long long)b * in_channels + ic) * in_depth + in_d) * in_height * in_width + in_h * in_width + in_w;
                                long long w_idx = (((long long)ic * out_channels + oc) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                                sum += x[x_idx] * weight[w_idx];
                            }
                        }
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        y[i] = sum;
    }
}

extern "C" void conv_transpose3d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int batch, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_d, int kernel_h, int kernel_w,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int out_depth, int out_height, int out_width,
    int output_pad_d, int output_pad_h, int output_pad_w,
    cudaStream_t stream)
{
    int total = batch * out_channels * out_depth * out_height * out_width;
    if (total <= 0) return;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    conv_transpose3d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y,
        batch, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_d, kernel_h, kernel_w,
        pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
        out_depth, out_height, out_width,
        output_pad_d, output_pad_h, output_pad_w);
}
