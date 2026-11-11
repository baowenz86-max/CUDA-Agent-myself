#include <cuda_runtime.h>

// CUDA kernel for ReplicationPad2d on NCHW layout.
// Input may have arbitrary strides; output is assumed contiguous with standard NCHW strides.
// Each thread computes one output element; grid-stride loop handles arbitrary size.
__global__ void replication_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_top,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_height,
    int64_t stride_width,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t step = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += step) {
        // Decompose linear index into output NCHW coordinates.
        int64_t w_out = i % W_out;
        int64_t tmp = i / W_out;
        int64_t h_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Replication source coordinates (clamped to input boundaries).
        int64_t w_in = w_out - pad_left;
        if (w_in < 0) w_in = 0;
        if (w_in >= W_in) w_in = W_in - 1;

        int64_t h_in = h_out - pad_top;
        if (h_in < 0) h_in = 0;
        if (h_in >= H_in) h_in = H_in - 1;

        // Gather from input using provided strides.
        int64_t src_idx = n * stride_batch + c * stride_channel
                        + h_in * stride_height + w_in * stride_width;
        output[i] = input[src_idx];
    }
}

// Launcher for replication_pad2d_forward.
// All parameters are passed as raw pointers or integers, no torch dependencies.
extern "C" void replication_pad2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_top,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_height,
    int64_t stride_width,
    cudaStream_t stream)
{
    int64_t total_elements = N * C * H_out * W_out;
    if (total_elements == 0) return;

    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding max grid dimension, use grid-stride loop.
    const int64_t max_grid_y = 65535;
    if (grid_size > max_grid_y) {
        grid_size = max_grid_y;
    }

    replication_pad2d_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input,
        output,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_left,
        pad_top,
        stride_batch,
        stride_channel,
        stride_height,
        stride_width,
        total_elements);
}
