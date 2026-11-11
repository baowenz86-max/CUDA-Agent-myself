#include <cuda_runtime.h>
#include <math.h>

// Reflection padding kernel for 4D tensor (N, C, H, W) -> (N, C, Hout, Wout)
// Input is assumed contiguous (handled by binding layer).
// Output is also contiguous.
// Reflection rule: if index < 0 => -index; if index >= dim => 2*dim - index - 2;
// For padding sizes typically small relative to dimension.
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int pad_left, int pad_right, int pad_top, int pad_bottom)
{
    int total = N * C * Hout * Wout;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total; linear += stride) {
        int w_out = linear % Wout;
        int h_out = (linear / Wout) % Hout;
        int c = (linear / (Wout * Hout)) % C;
        int n = linear / (Wout * Hout * C);

        int h_in = h_out - pad_top;
        int w_in = w_out - pad_left;

        // reflect h
        if (h_in < 0) h_in = -h_in;
        else if (h_in >= H) h_in = 2 * H - h_in - 2;
        // reflect w
        if (w_in < 0) w_in = -w_in;
        else if (w_in >= W) w_in = 2 * W - w_in - 2;

        // ensure within bounds (should be fine for valid padding)
        if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) {
            output[linear] = 0.0f; // fallback, should not happen
            continue;
        }

        int input_idx = ((n * C + c) * H + h_in) * W + w_in;
        output[linear] = input[input_idx];
    }
}

// Clamp min kernel: output = max(input, min_value) (equivalent to torch.minimum?)
// torch.minimum(input, min_value) returns element-wise min(input, min_value).
// Correct: if min_value is scalar, torch.minimum(input, min_value) = element-wise min(input, min_value).
// For float, NaN propagates: fminf propagates NaN as torch does for minimum (NaN if either is NaN).
// Note: torch.minimum is element-wise minimum, not maximum. Check semantics:
// torch.minimum(input, min_value) => each element = min(input[i], min_value).
// Using fminf(a, b) returns the smaller of a and b (NaN propagation).
__global__ void clamp_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    float min_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        output[i] = fminf(input[i], min_value);
    }
}

// Launcher for reflection_pad2d_forward
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream)
{
    int Hout = H + pad_top + pad_bottom;
    int Wout = W + pad_left + pad_right;
    int total = N * C * Hout * Wout;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, Hout, Wout,
        pad_left, pad_right, pad_top, pad_bottom);
}

// Launcher for clamp_min_forward
extern "C" void clamp_min_forward_launcher(
    const float* input,
    float* output,
    int total,
    float min_value,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    clamp_min_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total, min_value);
}
