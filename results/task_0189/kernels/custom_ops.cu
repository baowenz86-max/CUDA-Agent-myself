#include <cuda_runtime.h>

// CUDA kernel for adaptive average pooling 3D to output size (O, O, O)
// Input: (N, C, D, H, W), Output: (N, C, O, O, O)
// O is fixed at compile time? No, passed as parameter output_size (int).
// We'll implement generic output_size but specialize for 2? We'll use runtime output_size.
// Each thread computes one output element.
// Strides allow non-contiguous input.
__global__ void adaptive_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    int O,
    int out_stride_n, int out_stride_c, int out_stride_d, int out_stride_h, int out_stride_w,
    int total_output_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_output_elements; i += stride) {
        // Decode output indices: n, c, od, oh, ow
        int ow = i % O;
        int tmp = i / O;
        int oh = tmp % O;
        tmp /= O;
        int od = tmp % O;
        tmp /= O;
        int c = tmp % C;
        int n = tmp / C;

        // Compute pooling window in input
        int d_start = (int)floor((double)od * D / O);
        int d_end   = (int)ceil((double)(od + 1) * D / O);
        int h_start = (int)floor((double)oh * H / O);
        int h_end   = (int)ceil((double)(oh + 1) * H / O);
        int w_start = (int)floor((double)ow * W / O);
        int w_end   = (int)ceil((double)(ow + 1) * W / O);

        double sum = 0.0;
        int count = 0;

        const float* base = input
            + (size_t)n * in_stride_n
            + (size_t)c * in_stride_c;

        for (int d = d_start; d < d_end; ++d) {
            for (int h = h_start; h < h_end; ++h) {
                for (int w = w_start; w < w_end; ++w) {
                    const float* ptr = base
                        + (size_t)d * in_stride_d
                        + (size_t)h * in_stride_h
                        + (size_t)w * in_stride_w;
                    sum += (double)*ptr;
                    ++count;
                }
            }
        }

        // Write average
        size_t out_idx = (size_t)n * out_stride_n
                       + (size_t)c * out_stride_c
                       + (size_t)od * out_stride_d
                       + (size_t)oh * out_stride_h
                       + (size_t)ow * out_stride_w;
        output[out_idx] = (float)(sum / count);
    }
}

extern "C" void adaptive_avg_pool3d_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    int O,
    int out_stride_n, int out_stride_c, int out_stride_d, int out_stride_h, int out_stride_w,
    cudaStream_t stream)
{
    int total_output_elements = N * C * O * O * O;
    // Use 256 threads per block, grid size to cover with grid-stride loop
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;
    // Cap grid size to avoid too many blocks (max 65535 in x for old arch, but on modern can be larger).
    // A common approach: grid_size = min(grid_size, 65535*8) or use multi-dim. We'll use simple cap at 2^31-1 and rely on grid-stride loop.
    // Actually gridDim.x max is 2^31-1, so no issue. But to be safe with scheduling, cap at something like 65535*4.
    if (grid_size > 65535 * 4) {
        grid_size = 65535 * 4;
    }
    adaptive_avg_pool3d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        O,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        total_output_elements);
}
