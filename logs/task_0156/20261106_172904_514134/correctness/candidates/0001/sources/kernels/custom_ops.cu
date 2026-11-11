// kernels/custom_ops.cu
// CUDA kernels for custom_unfold and custom_pad1d
// No torch/ATen/c10/pybind dependencies here.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// custom_unfold: extracts sliding local blocks as in torch.nn.Unfold
// Input:  x is a 4D tensor of shape [N, C, H, W] (float32)
// Output: out is a 3D tensor of shape [N, C * kernel_size * kernel_size, L]
// where L = out_h * out_w
// out_h = (H + 2*pad_h - dilation*(kernel_size-1) - 1) / stride + 1
// out_w = (W + 2*pad_w - dilation*(kernel_size-1) - 1) / stride + 1
//
// The input may be non-contiguous, so we use input strides.
// We assume x is a CUDA tensor with float32 data.
// -----------------------------------------------------------------------------

__global__ void custom_unfold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_size, int dilation, int pad_h, int pad_w, int stride,
    int out_h, int out_w,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    long long total_output_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;

    for (long long i = idx; i < total_output_elements; i += grid_stride) {
        // Decompose linear output index into (n, ck, l)
        long long n = i / out_stride_n;
        long long rem = i % out_stride_n;
        long long ck = rem / out_stride_c;
        long long l = rem % out_stride_c;

        // ck = c * kernel_size^2 + k_index
        int ks2 = kernel_size * kernel_size;
        int c = ck / ks2;
        int k = ck % ks2;

        // k decomposes to (ki, kj) with channel-last order as PyTorch uses:
        // PyTorch's Unfold output shape is [N, C * prod(kernel_size), L]
        // For each output element, the channel block index varies as:
        // c_out = c * ks2 + ki * kernel_size + kj
        // where ki and kj are kernel positions (row major kernel_size x kernel_size)
        int ki = k / kernel_size;
        int kj = k % kernel_size;

        // l decomposes into (oh, ow)
        int oh = l / out_w;
        int ow = l % out_w;

        // Input coordinates
        int ih = oh * stride - pad_h + ki * dilation;
        int iw = ow * stride - pad_w + kj * dilation;

        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            const float* input_ptr = input + n * in_stride_n + c * in_stride_c
                                    + ih * in_stride_h + iw * in_stride_w;
            val = *input_ptr;
        }

        output[i] = val;
    }
}

// Launcher for custom_unfold
// The launcher receives raw pointers and metadata; it does not perform any tensor checks.
// It is expected that the binding layer allocates output and validates inputs.
extern "C" void custom_unfold_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_size, int dilation, int pad_h, int pad_w, int stride,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream)
{
    int out_h = (H + 2 * pad_h - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (W + 2 * pad_w - dilation * (kernel_size - 1) - 1) / stride + 1;
    long long L = (long long)out_h * out_w;
    long long channels_out = (long long)C * kernel_size * kernel_size;
    long long total = (long long)N * channels_out * L;

    if (total == 0) return;

    const int block_size = 256;
    // Grid cap: use at most 65535 blocks in x dimension, but grid-stride loop allows smaller grid.
    long long blocks_needed = (total + block_size - 1) / block_size;
    int grid = (int)min((long long)65535, blocks_needed);

    custom_unfold_kernel<<<grid, block_size, 0, stream>>>(
        input, output,
        N, C, H, W,
        kernel_size, dilation, pad_h, pad_w, stride,
        out_h, out_w,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_l,
        total
    );
}

// -----------------------------------------------------------------------------
// custom_pad1d: pads last dimension with (pad_left, pad_right) zeros.
// Input:  x is a 3D tensor of shape [N, C, L] (float32)
// Output: out is a 3D tensor of shape [N, C, L + pad_left + pad_right]
// x may be non-contiguous.
// -----------------------------------------------------------------------------

__global__ void custom_pad1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int L,
    int pad_left, int pad_right,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    long long total_output_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    int L_out = L + pad_left + pad_right;

    for (long long i = idx; i < total_output_elements; i += grid_stride) {
        // Decompose into (n, c, l_out)
        long long n = i / out_stride_n;
        long long rem = i % out_stride_n;
        long long c = rem / out_stride_c;
        long long l_out = rem % out_stride_c;

        float val = 0.0f;
        if (l_out >= pad_left && l_out < pad_left + L) {
            int l_in = l_out - pad_left;
            const float* input_ptr = input + n * in_stride_n + c * in_stride_c + l_in * in_stride_l;
            val = *input_ptr;
        }
        output[i] = val;
    }
}

// Launcher for custom_pad1d
extern "C" void custom_pad1d_launcher(
    const float* input,
    float* output,
    int N, int C, int L,
    int pad_left, int pad_right,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream)
{
    int L_out = L + pad_left + pad_right;
    long long total = (long long)N * C * L_out;

    if (total == 0) return;

    const int block_size = 256;
    long long blocks_needed = (total + block_size - 1) / block_size;
    int grid = (int)min((long long)65535, blocks_needed);

    custom_pad1d_kernel<<<grid, block_size, 0, stream>>>(
        input, output,
        N, C, L,
        pad_left, pad_right,
        in_stride_n, in_stride_c, in_stride_l,
        out_stride_n, out_stride_c, out_stride_l,
        total
    );
}
