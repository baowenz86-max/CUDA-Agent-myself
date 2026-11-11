#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// AvgPool2d kernel: computes average over kernel_h x kernel_w window
// x: input tensor [N, C, H, W] contiguous float32
// out: output tensor [N, C, outH, outW] contiguous float32
// kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
// outH = floor((H + 2*padding_h - kernel_h)/stride_h + 1)
// outW = floor((W + 2*padding_w - kernel_w)/stride_w + 1)
__global__ void avgpool2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w)
{
    int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear_idx = idx; linear_idx < total; linear_idx += stride) {
        int ow = linear_idx % outW;
        int oh = (linear_idx / outW) % outH;
        int c = (linear_idx / (outW * outH)) % C;
        int n = linear_idx / (outW * outH * C);

        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh * stride_h - padding_h + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = ow * stride_w - padding_w + kw;
                if (iw < 0 || iw >= W) continue;
                int input_idx = ((n * C + c) * H + ih) * W + iw;
                sum += x[input_idx];
                ++count;
            }
        }
        // count should be > 0 for valid output positions
        if (count > 0) {
            out[linear_idx] = sum / static_cast<float>(count);
        } else {
            out[linear_idx] = 0.0f;
        }
    }
}

// Hardtanh kernel: clip elementwise to [min_val, max_val]
__global__ void hardtanh_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int total,
    float min_val,
    float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        float val = x[i];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        out[i] = val;
    }
}

// Launcher for avgpool2d_forward
// Arguments:
//   x: input pointer (float*)
//   out: output pointer (float*)
//   N, C, H, W: input dimensions
//   kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w: pooling parameters
//   stream: CUDA stream
// Returns output dimensions via outH, outW (must be preallocated by caller?)
// Actually, we compute outH, outW internally, but launcher only takes pointers and dims,
// and does not allocate. The binding will allocate output of correct shape.
// For simplicity, we pass output dimensions as arguments.
// The binding code will compute outH, outW and allocate output accordingly.
// Thus launcher signature: (x, out, N, C, H, W, outH, outW, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, stream)
extern "C" void avgpool2d_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream)
{
    int total = N * C * outH * outW;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid excessive blocks (max grid dimension is 2^31-1)
    const int max_grid_size = 65535; // safe for all GPUs
    if (grid_size > max_grid_size) grid_size = max_grid_size;

    avgpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, N, C, H, W, outH, outW,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
}

// Launcher for hardtanh_forward
// Arguments:
//   x: input pointer (float*)
//   out: output pointer (float*)
//   total: number of elements
//   min_val, max_val: clip bounds
//   stream: CUDA stream
extern "C" void hardtanh_launcher(
    const float* x,
    float* out,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    const int max_grid_size = 65535;
    if (grid_size > max_grid_size) grid_size = max_grid_size;

    hardtanh_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, total, min_val, max_val);
}

// Optional: we can add error checking after kernel launch, but launcher should return void.
// The binding can call cudaGetLastError to check.
