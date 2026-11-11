#include <cuda_runtime.h>

__global__ void pixel_unshuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W,
    int upscale_factor,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t in_h_stride,
    int64_t in_w_stride,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    int upscale_sq = upscale_factor * upscale_factor;
    for (int64_t linear = idx; linear < total_elements; linear += grid_size) {
        // output is contiguous in N, C_out, H_out, W_out order
        // output shape: [N, C_out, H_out, W_out], C_out = C * upscale_sq
        int W_out = W / upscale_factor;
        int H_out = H / upscale_factor;
        int w_out = linear % W_out;
        int h_out = (linear / W_out) % H_out;
        int c_out = (linear / (W_out * H_out)) % (C * upscale_sq);
        int n = linear / (W_out * H_out * C * upscale_sq);

        // PyTorch pixel_unshuffle: output channel = c * upscale_sq + dy * upscale_factor + dx
        // So invert: c = c_out / upscale_sq, rem = c_out % upscale_sq, dy = rem / upscale_factor, dx = rem % upscale_factor
        int c_in = c_out / upscale_sq;
        int rem = c_out % upscale_sq;
        int dy = rem / upscale_factor;
        int dx = rem % upscale_factor;

        int h_in = h_out * upscale_factor + dy;
        int w_in = w_out * upscale_factor + dx;

        int64_t input_index = n * in_n_stride + c_in * in_c_stride + h_in * in_h_stride + w_in * in_w_stride;
        output[linear] = input[input_index];
    }
}

extern "C" void pixel_unshuffle_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int upscale_factor,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t in_h_stride,
    int64_t in_w_stride,
    cudaStream_t stream
) {
    int C_out = C * upscale_factor * upscale_factor;
    int H_out = H / upscale_factor;
    int W_out = W / upscale_factor;
    int64_t total_elements = (int64_t)N * C_out * H_out * W_out;

    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    // Cap grid to avoid exceeding max grid dimension; grid-stride loop handles remainder
    if (grid_size > 65535) grid_size = 65535;
    if (grid_size < 1) grid_size = 1;

    pixel_unshuffle_kernel<<<grid_size, block_size, 0, stream>>>(
        input,
        output,
        N,
        C,
        H,
        W,
        upscale_factor,
        in_n_stride,
        in_c_stride,
        in_h_stride,
        in_w_stride,
        total_elements
    );
}
