#include <cuda_runtime.h>
#include <math.h>

// Elementwise acos kernel for float tensors.
// Each thread processes one element, grid-stride loop ensures all elements are covered.
__global__ void acos_kernel(const float* __restrict__ input, float* __restrict__ output, long long numel) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = acosf(input[idx]);
    }
}

// Reflection padding kernel for 4D tensors (N, C, H, W).
// Treat each (n,c) plane independently, padding p on each side.
// Output dimensions: H_out = H + 2*p, W_out = W + 2*p.
// Each thread computes one output element.
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out, int p) {

    long long total = static_cast<long long>(N) * C * H_out * W_out;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    for (; idx < total; idx += stride) {
        // Decompose linear index: idx = (((n*C + c)*H_out + h_out)*W_out + w_out)
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Reflect coordinates: output coordinate 0..H_out-1 maps to input 0..H-1
        // With padding p, output index j corresponds to input index j-p.
        // Function reflect(j, size): if j<0 -> -j; if j>=size -> 2*size-2-j
        int h_in = h_out - p;
        if (h_in < 0) h_in = -h_in;
        else if (h_in >= H) h_in = 2 * H - 2 - h_in;

        int w_in = w_out - p;
        if (w_in < 0) w_in = -w_in;
        else if (w_in >= W) w_in = 2 * W - 2 - w_in;

        long long input_idx = ((static_cast<long long>(n) * C + c) * H + h_in) * W + w_in;
        output[idx] = input[input_idx];
    }
}

// Launcher for custom_acos. Input and output are contiguous float tensors of same shape.
extern "C" void custom_acos_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream) {

    int block_size = 256;
    long long blocks_ll = (numel + block_size - 1) / block_size;
    // Cap grid to avoid exceeding limits; grid-stride loop handles the rest.
    int max_blocks = 65535;
    int blocks = static_cast<int>(blocks_ll < max_blocks ? blocks_ll : max_blocks);
    acos_kernel<<<blocks, block_size, 0, stream>>>(input, output, numel);
}

// Launcher for custom_reflection_pad2d. Input and output are contiguous float tensors.
// padding is the symmetric padding on each side (total 2*p added per dimension).
extern "C" void custom_reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int padding,
    cudaStream_t stream) {

    int H_out = H + 2 * padding;
    int W_out = W + 2 * padding;
    long long total = static_cast<long long>(N) * C * H_out * W_out;

    int block_size = 256;
    long long blocks_ll = (total + block_size - 1) / block_size;
    int max_blocks = 65535;
    int blocks = static_cast<int>(blocks_ll < max_blocks ? blocks_ll : max_blocks);
    reflection_pad2d_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, padding);
}
