#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// PReLU forward kernel
// input: (N, C, H, W) float32
// weight: (C,) float32
// output: (N, C, H, W) float32
// -----------------------------------------------------------------------------
__global__ void prelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t channel_stride,   // elements between consecutive channels (H*W for contiguous)
    int64_t channels) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t c = (i / channel_stride) % channels;
        float w = weight[c];
        float val = input[i];
        output[i] = (val > 0.0f) ? val : (w * val);
    }
}

// -----------------------------------------------------------------------------
// Bilinear interpolation forward kernel (align_corners=false)
// input: (N, C, H_in, W_in) float32, contiguous
// output: (N, C, H_out, W_out) float32
// scale_factor: float, output spatial dims are computed as floor(scale * input)
// -----------------------------------------------------------------------------
__global__ void bilinear_interpolate_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    float scale_h,
    float scale_w) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output_elements = N * C * H_out * W_out;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_output_elements; i += stride) {
        // Decode linear index to (n, c, h_out, w_out)
        int64_t tmp = i;
        int64_t w_out = tmp % W_out;
        tmp /= W_out;
        int64_t h_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Input coordinates (align_corners=false)
        float h_src = (h_out + 0.5f) / scale_h - 0.5f;
        float w_src = (w_out + 0.5f) / scale_w - 0.5f;

        // Boundary handling: if h_src or w_src is -0.5, it corresponds to -0.5
        // but interpolation uses floor and ceil with clamping.
        int64_t h0 = (int64_t)floorf(h_src);
        int64_t w0 = (int64_t)floorf(w_src);
        int64_t h1 = h0 + 1;
        int64_t w1 = w0 + 1;

        float h_lambda = h_src - floorf(h_src);
        float w_lambda = w_src - floorf(w_src);

        // Clamp indices to [0, H_in-1] / [0, W_in-1]
        h0 = max(h0, (int64_t)0);
        h0 = min(h0, H_in - 1);
        h1 = max(h1, (int64_t)0);
        h1 = min(h1, H_in - 1);
        w0 = max(w0, (int64_t)0);
        w0 = min(w0, W_in - 1);
        w1 = max(w1, (int64_t)0);
        w1 = min(w1, W_in - 1);

        // Base offset for this (n, c) in input
        int64_t base = (n * C + c) * H_in * W_in;

        // Load four corner values
        float v00 = input[base + h0 * W_in + w0];
        float v01 = input[base + h0 * W_in + w1];
        float v10 = input[base + h1 * W_in + w0];
        float v11 = input[base + h1 * W_in + w1];

        // Bilinear interpolation
        float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
        float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
        float result = top * (1.0f - h_lambda) + bottom * h_lambda;

        output[i] = result;
    }
}

// -----------------------------------------------------------------------------
// Argmax over dimension 1 (channels) kernel
// input: (N, C, H, W) float32, contiguous
// output: (N, 1, H, W) int64, contiguous
// dim = 1, keepdim = true
// -----------------------------------------------------------------------------
__global__ void argmax_dim_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t spatial_size) {  // H*W

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_positions = N * spatial_size;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_positions; i += stride) {
        int64_t n = i / spatial_size;
        int64_t s = i % spatial_size;

        const float* base_ptr = input + n * C * spatial_size + s;
        float max_val = base_ptr[0];
        int64_t max_index = 0;

        for (int64_t c = 1; c < C; ++c) {
            float val = base_ptr[c * spatial_size];
            if (val > max_val) {
                max_val = val;
                max_index = c;
            }
        }

        // Output index: since keepdim=true, output has shape (N, 1, H, W),
        // but the spatial position mapping is the same.
        output[i] = max_index;
    }
}

// -----------------------------------------------------------------------------
// Host launcher functions (extern "C")
// -----------------------------------------------------------------------------

// prelu_forward(input_ptr, weight_ptr, output_ptr, total_elements, channel_stride, channels, stream)
extern "C" void prelu_forward_launcher(
    const float* input,
    const float* weight,
    float* output,
    int64_t total_elements,
    int64_t channel_stride,
    int64_t channels,
    cudaStream_t stream) {

    if (total_elements == 0) return;

    int threads = 256;
    int64_t blocks_64 = (total_elements + threads - 1) / threads;
    // Cap blocks at 65535, use grid-stride loop to cover remainder
    int blocks = (int)min(blocks_64, (int64_t)65535);

    prelu_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output, total_elements, channel_stride, channels);
}

// bilinear_interpolate_forward(input_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out, scale_h, scale_w, stream)
extern "C" void bilinear_interpolate_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    float scale_h,
    float scale_w,
    cudaStream_t stream) {

    int64_t total_output_elements = N * C * H_out * W_out;
    if (total_output_elements == 0) return;

    int threads = 256;
    int64_t blocks_64 = (total_output_elements + threads - 1) / threads;
    int blocks = (int)min(blocks_64, (int64_t)65535);

    bilinear_interpolate_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, scale_h, scale_w);
}

// argmax_dim_forward(input_ptr, output_ptr, N, C, spatial_size, stream)
extern "C" void argmax_dim_forward_launcher(
    const float* input,
    int64_t* output,
    int64_t N,
    int64_t C,
    int64_t spatial_size,
    cudaStream_t stream) {

    int64_t total_positions = N * spatial_size;
    if (total_positions == 0) return;

    int threads = 256;
    int64_t blocks_64 = (total_positions + threads - 1) / threads;
    int blocks = (int)min(blocks_64, (int64_t)65535);

    argmax_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, spatial_size);
}
