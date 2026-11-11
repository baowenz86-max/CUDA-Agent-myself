// CUDA kernels for upsampling_bilinear2d and constant_pad2d
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Kernel: bilinear upsampling (align_corners = false)
// Each thread computes one output pixel.
// Input is assumed to be contiguous (or at least row-major with stride provided).
// We pass input strides (or assume contiguous if stride==0).
// Grid-stride loop to handle large outputs.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void bilinear_upsample_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    bool align_corners,
    long long in_N_stride,
    long long in_C_stride,
    long long in_H_stride,
    long long in_W_stride)
{
    int total = N * C * H_out * W_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);

        // compute source coordinates
        float h_in_f, w_in_f;
        if (align_corners && H_out > 1) {
            h_in_f = (float)h_out * (H_in - 1) / (float)(H_out - 1);
        } else {
            h_in_f = ((float)h_out + 0.5f) / scale_factor - 0.5f;
        }
        if (align_corners && W_out > 1) {
            w_in_f = (float)w_out * (W_in - 1) / (float)(W_out - 1);
        } else {
            w_in_f = ((float)w_out + 0.5f) / scale_factor - 0.5f;
        }

        // clamp to [0, H_in-1] and [0, W_in-1]
        h_in_f = fmaxf(0.0f, h_in_f);
        w_in_f = fmaxf(0.0f, w_in_f);
        float h_floor_f = floorf(h_in_f);
        float w_floor_f = floorf(w_in_f);
        int h0 = (int)h_floor_f;
        int w0 = (int)w_floor_f;
        int h1 = min(h0 + 1, H_in - 1);
        int w1 = min(w0 + 1, W_in - 1);
        h0 = max(h0, 0);
        w0 = max(w0, 0);

        float h_lambda = h_in_f - h_floor_f;
        float w_lambda = w_in_f - w_floor_f;

        // base pointer for this n,c
        const scalar_t* in_base = input + n * in_N_stride + c * in_C_stride;

        // fetch four neighbors
        scalar_t v00 = in_base[h0 * in_H_stride + w0 * in_W_stride];
        scalar_t v01 = in_base[h0 * in_H_stride + w1 * in_W_stride];
        scalar_t v10 = in_base[h1 * in_H_stride + w0 * in_W_stride];
        scalar_t v11 = in_base[h1 * in_H_stride + w1 * in_W_stride];

        // bilinear interpolation
        float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
        float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
        float result = top * (1.0f - h_lambda) + bottom * h_lambda;

        output[idx] = (scalar_t)result;
    }
}

// -----------------------------------------------------------------------------
// Kernel: constant pad2d
// Each thread fills one output pixel (including padding area).
// Input strides are provided, can handle non-contiguous.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void constant_pad2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int left, int right, int top, int bottom,
    scalar_t value,
    long long in_N_stride,
    long long in_C_stride,
    long long in_H_stride,
    long long in_W_stride)
{
    int total = N * C * H_out * W_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);

        int w_in = w_out - left;
        int h_in = h_out - top;
        bool inside = (w_in >= 0 && w_in < W_in && h_in >= 0 && h_in < H_in);

        if (inside) {
            const scalar_t* in_base = input + n * in_N_stride + c * in_C_stride;
            output[idx] = in_base[h_in * in_H_stride + w_in * in_W_stride];
        } else {
            output[idx] = value;
        }
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" so C++ binding.cpp can call them)
// -----------------------------------------------------------------------------

// Bilinear upsampling launcher
// Args:
//   input: device pointer to input tensor (float)
//   output: device pointer to output tensor (float)
//   N, C, H_in, W_in: input dimensions
//   H_out, W_out: output dimensions
//   scale_factor: float scale
//   align_corners: bool
//   in_strides: array of 4 long long strides for N,C,H,W (or nullptr for contiguous assumption)
//   stream: CUDA stream
// Note: if in_strides is nullptr, we assume contiguous and compute strides as C*H*W, H*W, W, 1.
void upsample_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    bool align_corners,
    const long long* in_strides,
    cudaStream_t stream)
{
    long long sN, sC, sH, sW;
    if (in_strides) {
        sN = in_strides[0];
        sC = in_strides[1];
        sH = in_strides[2];
        sW = in_strides[3];
    } else {
        // contiguous assumed
        sN = (long long)C * H_in * W_in;
        sC = (long long)H_in * W_in;
        sH = W_in;
        sW = 1;
    }

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = min((total + threads - 1) / threads, 65535); // grid cap, grid-stride loop handles rest
    if (blocks <= 0) blocks = 1;

    bilinear_upsample_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        scale_factor, align_corners, sN, sC, sH, sW);
}

// Constant pad2d launcher
// Args:
//   input, output: device pointers
//   N, C, H_in, W_in
//   H_out, W_out
//   left, right, top, bottom: padding amounts
//   value: padding value
//   in_strides: array of 4 long long strides (or nullptr for contiguous)
//   stream
void constant_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int left, int right, int top, int bottom,
    float value,
    const long long* in_strides,
    cudaStream_t stream)
{
    long long sN, sC, sH, sW;
    if (in_strides) {
        sN = in_strides[0];
        sC = in_strides[1];
        sH = in_strides[2];
        sW = in_strides[3];
    } else {
        sN = (long long)C * H_in * W_in;
        sC = (long long)H_in * W_in;
        sH = W_in;
        sW = 1;
    }

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = min((total + threads - 1) / threads, 65535);
    if (blocks <= 0) blocks = 1;

    constant_pad2d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        left, right, top, bottom, value, sN, sC, sH, sW);
}

// Extern "C" wrapper names likely used by binding.cpp.
// The binding.cpp will likely define functions like:
// torch::Tensor upsampling_bilinear2d_forward(torch::Tensor x, double scale_factor, bool align_corners)
// and then call upsample_launcher(...)
// Similarly for constant_pad2d_forward.
// We provide these two launchers with clear names.

// Additional comment: We could also provide C-compatible names like upsampling_bilinear2d_launcher and constant_pad2d_launcher.
// But to avoid confusion, we'll define them as done above.
