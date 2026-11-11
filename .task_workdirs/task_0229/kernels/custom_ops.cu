#include <cuda_runtime.h>

// Kernel: zero padding for 4D tensor (N, C, H, W)
// Input may have arbitrary strides, output is contiguous with shape (N, C, H_out, W_out)
// pad_top, pad_bottom, pad_left, pad_right are non-negative ints
// Blocks: 2D grid over output width and height (max 65535 per dim, but output width/height small enough)
// Each thread handles one output element.
__global__ void zero_pad_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_top, int pad_left,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride)
{
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_h >= H_out || out_w >= W_out) return;

    // Determine corresponding input coordinates
    int in_h = out_h - pad_top;
    int in_w = out_w - pad_left;

    // Input index offset
    long long input_offset = 0;

    // For each batch and channel: need to loop over N and C? 
    // Instead we flatten N,C into blockIdx.z or use grid-stride? Since N*C can be large (48), use blockIdx.z to cover N*C.
    // We'll use blockIdx.z to index n and c combined, or better use linear block index and compute n,c inside.
    int nc = blockIdx.z; // up to N*C, limited to 65535 for grid z, but N*C=48 ok.
    if (nc >= N * C) return;

    int n = nc / C;
    int c = nc % C;

    long long base_input = n * input_n_stride + c * input_c_stride;
    long long base_output = ((long long)n * C + c) * H_out * W_out;

    if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
        long long input_idx = base_input + in_h * input_h_stride + in_w * input_w_stride;
        output[base_output + (long long)out_h * W_out + out_w] = input[input_idx];
    } else {
        output[base_output + (long long)out_h * W_out + out_w] = 0.0f;
    }
}

// Launcher function (extern "C" to be callable from binding)
extern "C" void zero_pad_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream)
{
    // Kernel launch configuration: use 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C  // grid z maximum 65535; N*C = 48 safe
    );

    zero_pad_kernel<<<grid, block, 0, stream>>>(
        output, input, N, C, H_in, W_in, H_out, W_out,
        pad_top, pad_left,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride
    );
}
