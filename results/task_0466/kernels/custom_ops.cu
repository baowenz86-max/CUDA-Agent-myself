// CUDA kernel for 3D average pooling with kernel_size=(2,2,2), stride=(2,2,2)
#include <cuda_runtime.h>

// Kernel: each thread computes one output element (n, c, od, oh, ow)
// Input strides are provided so that any memory layout works.
__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    int out_D, int out_H, int out_W,
    long long total_output_elements)
{
    long long global_idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; global_idx < total_output_elements; global_idx += stride) {
        // Decode output index: ow fastest, then oh, od, c, n
        int ow = global_idx % out_W;
        long long tmp = global_idx / out_W;
        int oh = tmp % out_H;
        tmp /= out_H;
        int od = tmp % out_D;
        tmp /= out_D;
        int c = tmp % C;
        int n = tmp / C;

        // Corresponding input base coordinates (top-left-front of 2x2x2 block)
        int id = od * 2;
        int ih = oh * 2;
        int iw = ow * 2;

        // Compute linear index for each of the 8 input elements using strides
        const float* base_ptr = input + n * (long long)in_stride_n
                                      + c * (long long)in_stride_c
                                      + id * (long long)in_stride_d
                                      + ih * (long long)in_stride_h
                                      + iw * (long long)in_stride_w;

        // Sum all 8 elements
        float sum = 0.0f;
        sum += base_ptr[0];                        // (id, ih, iw)
        sum += base_ptr[in_stride_w];              // (id, ih, iw+1)
        sum += base_ptr[in_stride_h];              // (id, ih+1, iw)
        sum += base_ptr[in_stride_h + in_stride_w]; // (id, ih+1, iw+1)
        sum += base_ptr[in_stride_d];              // (id+1, ih, iw)
        sum += base_ptr[in_stride_d + in_stride_w]; // (id+1, ih, iw+1)
        sum += base_ptr[in_stride_d + in_stride_h]; // (id+1, ih+1, iw)
        sum += base_ptr[in_stride_d + in_stride_h + in_stride_w]; // (id+1, ih+1, iw+1)

        // Compute output linear index (contiguous output assumed)
        long long out_idx = ((long long)n * C + c) * ((long long)out_D * out_H * out_W)
                            + ((long long)od * out_H + oh) * out_W + ow;
        output[out_idx] = sum * 0.125f; // divide by 8
    }
}

// Launcher function (C interface)
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    int out_D, int out_H, int out_W,
    cudaStream_t stream)
{
    long long total_output_elements = (long long)N * C * out_D * out_H * out_W;

    // Choose block size and compute grid size
    int block_size = 256;
    long long grid_size_long = (total_output_elements + block_size - 1) / block_size;
    // Limit grid size to max grid dimension (2^31-1)
    int max_grid = 2147483647;
    int grid_size = (grid_size_long > max_grid) ? max_grid : (int)grid_size_long;

    avg_pool3d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
        out_D, out_H, out_W,
        total_output_elements);
}
