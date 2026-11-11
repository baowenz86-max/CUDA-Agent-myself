#include <cuda_runtime.h>

// LeakyReLU kernel: out = x >= 0 ? x : negative_slope * x
template <typename T>
__global__ void leaky_relu_kernel(T* out, const T* in, float negative_slope, int64_t total_elements) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        T val = in[idx];
        out[idx] = val >= (T)0 ? val : (T)(negative_slope * (float)val);
    }
}

// AvgPool2d kernel: output[n, c, oh, ow] = average over valid input elements in window
// Supports arbitrary strides via input strides and sizes.
// This version uses double accumulation for improved numerical accuracy,
// and processes each output element with a single thread.
template <typename T>
__global__ void avg_pool2d_kernel(
    T* out,
    const T* in,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h_in,
    int64_t stride_w_in,
    int64_t stride_n_out,
    int64_t stride_c_out,
    int64_t stride_h_out,
    int64_t stride_w_out
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    // decode output index
    int64_t tmp = idx;
    int64_t ow = tmp % W_out;
    tmp /= W_out;
    int64_t oh = tmp % H_out;
    tmp /= H_out;
    int64_t c = tmp % C;
    tmp /= C;
    int64_t n = tmp;

    // input window start
    int64_t in_start_h = oh * stride_h - pad_h;
    int64_t in_start_w = ow * stride_w - pad_w;

    // Accumulate in double to reduce floating-point error
    double sum = 0.0;
    int count = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        int64_t ih = in_start_h + kh;
        if (ih < 0 || ih >= H_in) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int64_t iw = in_start_w + kw;
            if (iw < 0 || iw >= W_in) continue;
            const T* in_ptr = in + n * stride_n + c * stride_c + ih * stride_h_in + iw * stride_w_in;
            sum += (double)(*in_ptr);
            ++count;
        }
    }

    // Compute average; count is guaranteed > 0
    T* out_ptr = out + n * stride_n_out + c * stride_c_out + oh * stride_h_out + ow * stride_w_out;
    *out_ptr = (T)(sum / (double)count);
}

// Launcher for leaky relu
extern "C" void custom_leaky_relu_launcher(
    void* out,
    const void* in,
    float negative_slope,
    int64_t total_elements,
    cudaStream_t stream
) {
    const int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    leaky_relu_kernel<float><<<(unsigned int)blocks, threads, 0, stream>>>(
        (float*)out, (const float*)in, negative_slope, total_elements
    );
}

// Launcher for avg_pool2d
extern "C" void custom_avg_pool2d_launcher(
    void* out,
    const void* in,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h_in,
    int64_t stride_w_in,
    int64_t stride_n_out,
    int64_t stride_c_out,
    int64_t stride_h_out,
    int64_t stride_w_out,
    cudaStream_t stream
) {
    const int threads = 256;
    int64_t total = N * C * H_out * W_out;
    int64_t blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<float><<<(unsigned int)blocks, threads, 0, stream>>>(
        (float*)out,
        (const float*)in,
        N, C, H_in, W_in, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        stride_n, stride_c, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore error; caller will check via cudaGetLastError after launch
    }
}
