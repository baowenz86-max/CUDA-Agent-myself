#include <cuda_runtime.h>

// Pixel unshuffle kernel
// output shape: [N, C*ds*ds, H/ds, W/ds]
// input shape:  [N, C, H, W]
// Index mapping: output[n, c*ds*ds + k*ds + j, h, w] = input[n, c, h*ds + k, w*ds + j]
// where k in [0, ds), j in [0, ds)
__global__ void pixel_unshuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W, int ds,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    long long total_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_elements; idx += stride) {
        // decompose idx into output coordinates (n, oc, oh, ow)
        int ow = idx % out_stride_w;
        int tmp = idx / out_stride_w;
        int oh = tmp % out_stride_h;
        tmp /= out_stride_h;
        int oc = tmp % out_stride_c;
        int n = tmp / out_stride_c;

        // oc = c*ds*ds + k*ds + j
        int c = oc / (ds * ds);
        int rem = oc % (ds * ds);
        int k = rem / ds;
        int j = rem % ds;

        // input coordinates
        int ih = oh * ds + k;
        int iw = ow * ds + j;

        // compute input offset using strides
        long long in_idx = (long long)n * in_stride_n + (long long)c * in_stride_c
                           + (long long)ih * in_stride_h + (long long)iw * in_stride_w;

        // output offset using strides
        long long out_idx = (long long)n * out_stride_n + (long long)oc * out_stride_c
                            + (long long)oh * out_stride_h + (long long)ow * out_stride_w;

        output[out_idx] = input[in_idx];
    }
}

// Launcher: pixel_unshuffle_forward_launcher
// Parameters:
//   input: pointer to input float data (device)
//   output: pointer to output float data (device)
//   N, C, H, W: input dimensions
//   ds: downscale factor (upscale_factor)
//   in_strides: array of 4 ints (stride for N,C,H,W) of input
//   out_strides: array of 4 ints (stride for N,C,H,W) of output
//   stream: CUDA stream
// Implements pixel_unshuffle semantics.
extern "C" void pixel_unshuffle_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W, int ds,
    const int* in_strides,
    const int* out_strides,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || ds <= 0) {
        return;
    }

    int out_H = H / ds;
    int out_W = W / ds;
    int out_C = C * ds * ds;
    long long total_elements = (long long)N * out_C * out_H * out_W;

    if (total_elements == 0) {
        return;
    }

    // Choose block size and grid size
    int block_size = 256;
    // Use grid-stride loop with a reasonable number of blocks
    int max_blocks = 65535; // maximum grid dimension x for older GPUs
    long long desired_blocks = (total_elements + block_size - 1) / block_size;
    int grid_size = (int)min((long long)max_blocks, desired_blocks);
    if (grid_size == 0) grid_size = 1;

    pixel_unshuffle_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, ds,
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        total_elements);
}
