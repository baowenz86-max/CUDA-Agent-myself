// kernels/custom_ops.cu
// CUDA kernels and launchers for custom_pad and custom_max_dim.
// No torch/ATen/c10/pybind dependencies.

#include <cuda_runtime.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// custom_pad kernel: ZeroPad2d on 4D tensor
// Input x: [N, C, H_in, W_in]
// Padding: left, right, top, bottom
// Output y: [N, C, H_out, W_out], H_out = H_in + top + bottom, W_out = W_in + left + right
// Supports arbitrary strides.
// Each thread computes one output element (with grid-stride loop if needed).
// -----------------------------------------------------------------------------
__global__ void custom_pad_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_top,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_h,
    int64_t x_stride_w,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // Decompose linear index into (n, c, oh, ow)
        int64_t ow = linear % W_out;
        int64_t tmp = linear / W_out;
        int64_t oh = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Map output coordinates to input coordinates
        int64_t ih = oh - pad_top;
        int64_t iw = ow - pad_left;

        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
            int64_t input_offset = n * x_stride_n + c * x_stride_c + ih * x_stride_h + iw * x_stride_w;
            y[linear] = x[input_offset];
        } else {
            y[linear] = 0.0f;
        }
    }
}

// Launcher for custom_pad (no tensor allocation, just fill output)
extern "C" void custom_pad_launcher(
    const float* x,
    float* y,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_h,
    int64_t x_stride_w,
    cudaStream_t stream)
{
    int64_t total_elements = N * C * H_out * W_out;
    if (total_elements == 0) return;

    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 2147483647LL) { // grid dimension limit for x
        blocks = 2147483647LL;
    }

    custom_pad_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        x, y, N, C, H_in, W_in, H_out, W_out,
        pad_left, pad_top,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        total_elements);
}

// -----------------------------------------------------------------------------
// custom_max_dim kernel: max over channels (dim=1) for 4D tensor [N, C, H, W]
// Output shape [N, H, W] = max over C.
// Assumes C > 0.
// Each thread handles one output element (n, h, w), loops over channels.
// Uses simple maximum propagation, matches torch.max semantics for NaN? 
// torch.max with NaN: returns NaN if any NaN present (IEEE max semantics).
// We'll use fmaxf, which returns NaN if one operand is NaN (actually fmaxf returns the non-NaN operand if one is NaN due to C99? No, fmaxf propagates NaN in CUDA? 
// CUDA fmaxf(a,b) returns the numeric maximum, and if one is NaN, returns the other? Let's verify: CUDA math API follows IEEE-754: fmaxf returns the larger of two values, and if one is NaN, returns the other (non-NaN) value. torch.max behavior: torch.max returns NaN if any input is NaN? Actually torch.max(input, dim) returns the maximum value; if there are NaNs, the result may be NaN depending on input? PyTorch torch.max documentation: "If there are multiple maximal values then the indices of the first maximal value are returned". Regarding NaN: torch.max propagates NaN? Let's check: torch.max(torch.tensor([1., float('nan')])) returns tensor(nan). So torch.max propagates NaN. Our kernel must match that: if any NaN, output NaN. fmaxf in CUDA would return the non-NaN operand, which is wrong. So we need to implement a comparison that propagates NaN: use (a > b) ? a : b, but if a or b is NaN, that returns b or a? For NaN, a > b is false, so returns b. We need to explicitly propagate NaN. Could use isnan checks or use fmaxf with additional checks? Simpler: use a loop where we keep current_max; if current_max is NaN, it should stay NaN? Actually torch.max returns NaN if any input is NaN. We can implement: if (b != b) { max = b; } else if (a != a) { max = a; } else if (a > b) max = a; else max = b; that propagates first NaN encountered? But torch.max returns NaN if any NaN exists, regardless of order. So we can just track "has_nan" flag and final result NaN. But for simplicity and given input is randn (no NaN expected), we can use fmaxf which gives non-NaN result if one NaN. However, contract says dtype float32 and values could theoretically include NaN but reference torch.max would produce NaN. We'll implement a kernel that matches torch.max semantics for ordinary numbers, and for NaN we can use fmaxf? Actually we need to be safe. Let's implement a simple loop that uses (val > max) ? val : max; if val is NaN, comparison false, so max remains previous, which may not be NaN. That would not propagate NaN. To propagate NaN, we can add a bool flag per thread: bool seen_nan = false; during loop, if isnan(val) seen_nan = true; after loop, if seen_nan output = nan. That matches torch.max returning NaN if any NaN present. We'll do that.
// -----------------------------------------------------------------------------
__global__ void custom_max_dim_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_h,
    int64_t x_stride_w,
    int64_t total_elements_y)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t linear = idx; linear < total_elements_y; linear += stride) {
        // Decompose linear index into (n, h, w)
        int64_t w = linear % W;
        int64_t tmp = linear / W;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        float max_val = -INFINITY;
        bool seen_nan = false;
        for (int64_t c = 0; c < C; ++c) {
            int64_t offset = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w;
            float val = x[offset];
            if (isnan(val)) {
                seen_nan = true;
            } else if (val > max_val) {
                max_val = val;
            }
        }
        y[linear] = seen_nan ? nanf("") : max_val;
    }
}

// Launcher for custom_max_dim (dim=1)
extern "C" void custom_max_dim_launcher(
    const float* x,
    float* y,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_h,
    int64_t x_stride_w,
    cudaStream_t stream)
{
    int64_t total_elements_y = N * H * W;
    if (total_elements_y == 0 || C == 0) return;

    const int block_size = 256;
    int64_t blocks = (total_elements_y + block_size - 1) / block_size;
    if (blocks > 2147483647LL) {
        blocks = 2147483647LL;
    }

    custom_max_dim_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        x, y, N, C, H, W,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        total_elements_y);
}
