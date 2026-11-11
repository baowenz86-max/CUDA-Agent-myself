#include <cuda_runtime.h>
#include <stdint.h>

// Softmax along dim=1 (channels) for each spatial location (n, h, w)
// x: input tensor (N, C, H, W) with arbitrary strides;
// y: output tensor same shape; out_strides can be same as in_strides or chosen by binding.
__global__ void softmax_dim1_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int64_t N, int64_t C, int64_t H, int64_t W,
    const int64_t* __restrict__ in_stride,  // 4 strides: N, C, H, W
    const int64_t* __restrict__ out_stride
) {
    int64_t total_spatial = N * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t spatial = idx; spatial < total_spatial; spatial += stride) {
        // Decode spatial index into n, h, w
        int64_t n = spatial / (H * W);
        int64_t rem = spatial - n * (H * W);
        int64_t h = rem / W;
        int64_t w = rem - h * W;

        const float* x_base = x + n * in_stride[0] + h * in_stride[2] + w * in_stride[3];
        float* y_base = y + n * out_stride[0] + h * out_stride[2] + w * out_stride[3];

        // First pass: find max value for numerical stability
        float max_val = -__FLT_MAX__;
        for (int64_t c = 0; c < C; ++c) {
            float val = x_base[c * in_stride[1]];
            if (val > max_val) max_val = val;
        }

        // Second pass: compute sum of exp(x - max)
        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            float val = __expf(x_base[c * in_stride[1]] - max_val);
            sum += val;
        }

        // Third pass: write normalized values
        float inv_sum = 1.0f / sum;
        for (int64_t c = 0; c < C; ++c) {
            float val = __expf(x_base[c * in_stride[1]] - max_val);
            y_base[c * out_stride[1]] = val * inv_sum;
        }
    }
}

// Constant pad 2D: pad left, right, top, bottom with constant value
// x: input (N,C,H,W) with in_stride; y: output (N,C,H+top+bottom, W+left+right) with out_stride
__global__ void constant_pad2d_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_l, int64_t pad_r, int64_t pad_t, int64_t pad_b,
    float value,
    const int64_t* __restrict__ in_stride,
    const int64_t* __restrict__ out_stride
) {
    int64_t H_out = H_in + pad_t + pad_b;
    int64_t W_out = W_in + pad_l + pad_r;
    int64_t total = N * C * H_out * W_out;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total; linear += stride) {
        // Decode output index: n, c, h_out, w_out
        int64_t n = linear / (C * H_out * W_out);
        int64_t rem = linear - n * (C * H_out * W_out);
        int64_t c = rem / (H_out * W_out);
        rem -= c * (H_out * W_out);
        int64_t h_out = rem / W_out;
        int64_t w_out = rem - h_out * W_out;

        bool inside = (h_out >= pad_t && h_out < pad_t + H_in && 
                       w_out >= pad_l && w_out < pad_l + W_in);
        float out_val = value;
        if (inside) {
            int64_t h_in = h_out - pad_t;
            int64_t w_in = w_out - pad_l;
            const float* x_ptr = x + n * in_stride[0] + c * in_stride[1] + h_in * in_stride[2] + w_in * in_stride[3];
            out_val = *x_ptr;
        }
        float* y_ptr = y + n * out_stride[0] + c * out_stride[1] + h_out * out_stride[2] + w_out * out_stride[3];
        *y_ptr = out_val;
    }
}

extern "C" {

// Launcher for softmax dim=1
// Parameters:
//   y: output tensor pointer (float*)
//   x: input tensor pointer (const float*)
//   N, C, H, W: tensor dimensions
//   in_stride: int64_t array of 4 strides for x
//   out_stride: int64_t array of 4 strides for y
//   stream: CUDA stream
void softmax_dim1_forward(
    float* y,
    const float* x,
    int64_t N, int64_t C, int64_t H, int64_t W,
    const int64_t* in_stride,
    const int64_t* out_stride,
    cudaStream_t stream)
{
    int64_t total_spatial = N * H * W;
    if (total_spatial <= 0) return;
    int threads = 256;
    int64_t blocks_64 = (total_spatial + threads - 1) / threads;
    // Cap grid size to avoid exceeding limits; use grid-stride loop in kernel.
    const int64_t max_grid_x = 2147483647LL;
    int64_t blocks = blocks_64 < max_grid_x ? blocks_64 : max_grid_x;
    softmax_dim1_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        y, x, N, C, H, W, in_stride, out_stride);
    // Error check is left to caller; here we only ensure launch did not fail synchronously.
    // cudaGetLastError();
}

// Launcher for constant pad2d
// Parameters:
//   y: output tensor pointer (float*)
//   x: input tensor pointer (const float*)
//   N, C, H_in, W_in: input dims
//   pad_l, pad_r, pad_t, pad_b: padding amounts
//   value: fill value
//   in_stride: int64_t array of 4 strides for x
//   out_stride: int64_t array of 4 strides for y
//   stream: CUDA stream
void constant_pad2d_forward(
    float* y,
    const float* x,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_l, int64_t pad_r, int64_t pad_t, int64_t pad_b,
    float value,
    const int64_t* in_stride,
    const int64_t* out_stride,
    cudaStream_t stream)
{
    int64_t H_out = H_in + pad_t + pad_b;
    int64_t W_out = W_in + pad_l + pad_r;
    int64_t total = N * C * H_out * W_out;
    if (total <= 0) return;
    int threads = 256;
    int64_t blocks_64 = (total + threads - 1) / threads;
    const int64_t max_grid_x = 2147483647LL;
    int64_t blocks = blocks_64 < max_grid_x ? blocks_64 : max_grid_x;
    constant_pad2d_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        y, x, N, C, H_in, W_in, pad_l, pad_r, pad_t, pad_b, value,
        in_stride, out_stride);
}

} // extern "C"
