#include <cuda_runtime.h>

// Reflection padding kernel for 4D float32 tensor with arbitrary strides.
// Output is contiguous with shape [N, C, H_out, W_out].
// Padding is symmetric: pad_top = pad_bottom = pad, pad_left = pad_right = pad.
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int pad,
    long long input_stride_n,   // stride in elements for N dimension (input is contiguous-like with possible strides)
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long total_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        tmp /= C;
        int n = tmp;

        // Compute input coordinates with reflection.
        // Reflection without repeating edge: for pad=3 and H=2, output indices:
        // h_in = 0: reflect 3 -> input index 1? Actually PyTorch ReflectionPad2d reflects excluding edge.
        // Formula: for output index i (0-based), input index = reflect_index(i - pad, H).
        // reflect_index(x, size): x in [0, size-1] normal; else mirror.
        // Using common formula: int ix = abs(x) % (2*size - 2); if ix >= size then ix = 2*size - 2 - ix;
        // More precisely: if x < 0: x = -x; if x >= size: x = 2*size - 2 - x;
        // However with pad=3, x = -3 -> abs=3, then if 3 >= H (say H=2), x = 2*2-2-3 = -1? Need loop.
        // Easier: use iterative reflection until in bounds:
        int h_in = h_out - pad;
        int w_in = w_out - pad;

        // Reflect h_in into [0, H-1]
        while (h_in < 0 || h_in >= H) {
            if (h_in < 0) h_in = -h_in;
            else if (h_in >= H) h_in = 2*H - 2 - h_in;
        }
        while (w_in < 0 || w_in >= W) {
            if (w_in < 0) w_in = -w_in;
            else if (w_in >= W) w_in = 2*W - 2 - w_in;
        }

        // For H=1 or W=1, the above may not converge (since 2*H-2 = 0). Handle edge case:
        if (H == 1) h_in = 0;
        if (W == 1) w_in = 0;

        // Compute input linear index using strides
        long long input_idx = (long long)n * input_stride_n +
                              (long long)c * input_stride_c +
                              (long long)h_in * input_stride_h +
                              (long long)w_in * input_stride_w;
        output[idx] = input[input_idx];
    }
}

// Add scalar elementwise: scalar tensor is contiguous [1,1,1,1], so scalar value is scalar_ptr[0].
__global__ void add_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scalar_ptr,
    long long total_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    float scalar = scalar_ptr[0];

    for (; idx < total_elements; idx += stride) {
        output[idx] = input[idx] + scalar;
    }
}

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int pad,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    // Grid size: enough to cover total_elements, but cap at a reasonable number; use grid-stride loop.
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // max grid dimension x
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, pad,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        total_elements);
}

extern "C" void add_scalar_launcher(
    const float* input,
    float* output,
    const float* scalar_ptr,
    long long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    add_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scalar_ptr, total_elements);
}
