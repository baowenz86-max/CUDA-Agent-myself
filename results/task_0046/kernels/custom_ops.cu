#include <cuda_runtime.h>
#include <stdint.h>

// Bilinear upsampling kernel for 4D tensor (N, C, H, W)
// Supports arbitrary strides; each thread processes one output element.
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t stride_n_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    float scale_factor) {
    int64_t total = N * C * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (; idx < total; idx += grid_size) {
        int64_t w_out = idx % W_out;
        int64_t tmp = idx / W_out;
        int64_t h_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        float h1 = (h_out + 0.5f) / scale_factor - 0.5f;
        float w1 = (w_out + 0.5f) / scale_factor - 0.5f;
        if (h1 < 0) h1 = 0;
        if (w1 < 0) w1 = 0;
        int64_t h0 = (int64_t)h1;
        int64_t w0 = (int64_t)w1;
        int64_t h1_int = h0 + 1;
        int64_t w1_int = w0 + 1;
        if (h0 >= H_in - 1) { h0 = H_in - 1; h1_int = H_in - 1; }
        if (w0 >= W_in - 1) { w0 = W_in - 1; w1_int = W_in - 1; }
        float h_lambda = h1 - h0;
        float w_lambda = w1 - w0;

        const float* base_in = input + n * stride_n_in + c * stride_c_in;
        int64_t i00 = h0 * stride_h_in + w0 * stride_w_in;
        int64_t i01 = h0 * stride_h_in + w1_int * stride_w_in;
        int64_t i10 = h1_int * stride_h_in + w0 * stride_w_in;
        int64_t i11 = h1_int * stride_h_in + w1_int * stride_w_in;

        float top = base_in[i00] * (1.0f - w_lambda) + base_in[i01] * w_lambda;
        float bottom = base_in[i10] * (1.0f - w_lambda) + base_in[i11] * w_lambda;
        float val = top * (1.0f - h_lambda) + bottom * h_lambda;

        int64_t out_idx = n * stride_n_out + c * stride_c_out + h_out * stride_h_out + w_out * stride_w_out;
        output[out_idx] = val;
    }
}

// Elementwise ReLU kernel for arbitrary 1D flattened indexing (strides handled by offset calculation in launcher?)
// Actually, we compute the linear index and access output[index] and input[index] assuming contiguous flattened memory.
// The launcher will ensure the input and output are contiguous for simplicity, or we can pass strides for 4D.
// Since relu_where is applied on the output of interpolate which is contiguous, we assume contiguous.
__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (; idx < total; idx += grid_size) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

// Host launcher for bilinear upsampling (no PyTorch dependencies)
extern "C" void interpolate_bilinear_scale_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t stride_n_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    float scale_factor,
    cudaStream_t stream) {
    int64_t total = N * C * H_out * W_out;
    int block_size = 256;
    int64_t blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535; // grid cap, grid-stride loop handles rest
    bilinear_upsample_kernel<<<(int)blocks, block_size, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out,
        scale_factor);
}

// Host launcher for ReLU (assumes contiguous input and output)
extern "C" void relu_where_launcher(
    const float* input,
    float* output,
    int64_t total,
    cudaStream_t stream) {
    int block_size = 256;
    int64_t blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    relu_kernel<<<(int)blocks, block_size, 0, stream>>>(input, output, total);
}
