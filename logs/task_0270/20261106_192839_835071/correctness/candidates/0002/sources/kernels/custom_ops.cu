#include <cuda_runtime.h>

// Compute x * (pi / 180) for each element
__global__ void deg2rad_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const float scale = 0.017453292519943295f; // pi / 180
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] * scale;
    }
}

// Reflection padding for last two dimensions (H, W)
// output: [N, C, H_out, W_out], input: [N, C, H_in, W_in]
// pad_l, pad_r, pad_t, pad_b are non-negative
__global__ void reflection_pad2d_kernel(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int pad_t,
    int pad_l,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t out_stride = blockDim.x * gridDim.x;
    int64_t total_elements = N * C * H_out * W_out;

    // Reflection helper: maps coordinate to [0, size-1] with reflection
    auto reflect = [](int64_t coord, int64_t size) -> int64_t {
        if (size == 1) return 0;
        coord = coord % (2 * size - 2);
        if (coord < 0) coord += 2 * size - 2;
        if (coord < size) return coord;
        return 2 * size - 2 - coord;
    };

    for (int64_t idx = out_idx; idx < total_elements; idx += out_stride) {
        int64_t w_out = idx % W_out;
        int64_t tmp = idx / W_out;
        int64_t h_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Compute input coordinates
        int64_t h_in = h_out - pad_t;
        int64_t w_in = w_out - pad_l;

        // Reflect if outside
        h_in = reflect(h_in, H_in);
        w_in = reflect(w_in, W_in);

        int64_t in_idx = n * in_stride_n + c * in_stride_c + h_in * in_stride_h + w_in * in_stride_w;
        output[idx] = input[in_idx];
    }
}

// Launcher for custom_deg2rad
extern "C" void custom_deg2rad_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Use grid-stride loop for large data
    if (grid_size > 65535) grid_size = 65535;
    deg2rad_kernel<<<grid_size, block_size, 0, stream>>>(output, input, numel);
}

// Launcher for custom_reflection_pad2d
extern "C" void custom_reflection_pad2d_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int pad_t,
    int pad_l,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    cudaStream_t stream
) {
    int64_t total_elements = N * C * H_out * W_out;
    if (total_elements == 0) return;
    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    reflection_pad2d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, N, C, H_in, W_in, H_out, W_out, pad_t, pad_l,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w
    );
}
