#include <cuda_runtime.h>
#include <cstdint>

// Upsample bilinear2d kernel.
// Each thread computes one output element.
// Input can have arbitrary stride; output is assumed contiguous.
// align_corners is passed as bool.
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    bool align_corners,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H_out * W_out;
    if (idx >= total) return;

    // Decode output coordinates
    int w_out = idx % W_out;
    long long tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Compute source coordinates
    float src_h, src_w;
    if (align_corners) {
        src_h = (float)h_out * (H_in - 1) / (float)(H_out - 1);
        src_w = (float)w_out * (W_in - 1) / (float)(W_out - 1);
    } else {
        src_h = ((float)h_out + 0.5f) / scale_factor - 0.5f;
        src_w = ((float)w_out + 0.5f) / scale_factor - 0.5f;
    }

    // Clamp to [0, H_in-1] and [0, W_in-1]
    src_h = fmaxf(0.0f, fminf(src_h, (float)(H_in - 1)));
    src_w = fmaxf(0.0f, fminf(src_w, (float)(W_in - 1)));

    int h0 = (int)floorf(src_h);
    int w0 = (int)floorf(src_w);
    int h1 = min(h0 + 1, H_in - 1);
    int w1 = min(w0 + 1, W_in - 1);

    float h_lambda = src_h - (float)h0;
    float w_lambda = src_w - (float)w0;

    // Base pointer for this n,c
    const float* in_base = input + n * input_stride_n + c * input_stride_c;

    // Read four neighbors
    float v00 = in_base[h0 * input_stride_h + w0 * input_stride_w];
    float v01 = in_base[h0 * input_stride_h + w1 * input_stride_w];
    float v10 = in_base[h1 * input_stride_h + w0 * input_stride_w];
    float v11 = in_base[h1 * input_stride_h + w1 * input_stride_w];

    // Bilinear interpolation
    float val = (1.0f - h_lambda) * (1.0f - w_lambda) * v00 +
                h_lambda * (1.0f - w_lambda) * v10 +
                (1.0f - h_lambda) * w_lambda * v01 +
                h_lambda * w_lambda * v11;

    // Output index is contiguous: ((n*C + c)*H_out + h_out)*W_out + w_out
    long long out_idx = (((long long)n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = val;
}

// Logical AND kernel for (x > lower) && (x < upper)
// Input arbitrary stride, output bool contiguous.
__global__ void logical_and_gt_lt_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    long long numel,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    int ndims,
    long long* input_strides,  // device array of strides (length ndims)
    long long* input_sizes,    // device array of sizes (length ndims)
    float lower,
    float upper)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // For 4D only, decode coordinates and use strides directly.
    // We assume ndims == 4 as per contract.
    // To be safe, we provide a generic version but assume 4D.
    int w = idx % input_sizes[3];
    long long tmp = idx / input_sizes[3];
    int h = tmp % input_sizes[2];
    tmp /= input_sizes[2];
    int c = tmp % input_sizes[1];
    int n = tmp / input_sizes[1];

    long long offset = n * input_strides[0] + c * input_strides[1] + h * input_strides[2] + w * input_strides[3];
    float val = input[offset];
    bool cond = (val > lower) && (val < upper);
    output[idx] = cond;
}

// Launcher for upsample_bilinear2d
// Input/output are raw pointers. Strides passed as int64.
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    bool align_corners,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        scale_factor, align_corners,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Cannot use printf in release, but launcher can ignore or handle later.
        // We assume binding will check.
    }
}

// Launcher for logical_and_gt_lt
// Input is expected 4D contiguous or with strides provided.
extern "C" void logical_and_gt_lt_launcher(
    const float* input,
    bool* output,
    long long numel,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    float lower,
    float upper,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    logical_and_gt_lt_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        4, nullptr, nullptr,  // not used
        lower, upper);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore for now
    }
}
