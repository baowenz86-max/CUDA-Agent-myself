#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// AvgPool2d forward kernel
// -----------------------------------------------------------------------------
// Computes average pooling with kernel_size=3, stride=2, padding=1, count_include_pad=True
// Input: (N, C, H, W) float32, output: (N, C, H_out, W_out)
// H_out = floor((H + 2*padding - kernel_size)/stride) + 1
// W_out = floor((W + 2*padding - kernel_size)/stride) + 1
// Each output element is average of 3x3 input window (with zero padding outside)
// Denominator is always kernel_size*kernel_size = 9 (count_include_pad=True)
__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding
) {
    int total_elements = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int linear_idx = idx; linear_idx < total_elements; linear_idx += grid_stride) {
        // Decode linear index: (n, c, h_out, w_out)
        int w_out = linear_idx % W_out;
        int tmp = linear_idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Input window origin in input coordinates
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_start + kh;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = w_start + kw;
                if (w_in < 0 || w_in >= W) continue;
                // Input is contiguous (assume contiguous for simplicity; binding layer will ensure)
                sum += input[((n * C + c) * H + h_in) * W + w_in];
            }
        }
        float denom = (float)(kernel_size * kernel_size); // 9.0f
        int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
        output[out_idx] = sum / denom;
    }
}

// -----------------------------------------------------------------------------
// Pow scalar forward kernel
// -----------------------------------------------------------------------------
// Computes element-wise power: output[i] = input[i] ^ exponent
// exponent is passed as a pointer to a single float value (0-dim tensor)
__global__ void pow_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ exponent_ptr,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    float exponent = *exponent_ptr;

    for (int i = idx; i < total_elements; i += grid_stride) {
        output[i] = powf(input[i], exponent);
    }
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C")
// -----------------------------------------------------------------------------
// These are called from binding.cpp. All pointers are raw device pointers.
// Input tensors are assumed to be CUDA float32. Output tensors are pre-allocated by binding layer.
// stream: CUDA stream on which to launch kernels.

// Launch avgpool2d_forward
// Parameters:
//   input: device pointer to input tensor (N,C,H,W) contiguous float32
//   output: device pointer to output tensor (N,C,H_out,W_out) contiguous float32 (allocated by caller)
//   N, C, H, W: input dimensions
//   H_out, W_out: output spatial dimensions
//   kernel_size, stride, padding: pooling parameters
//   stream: CUDA stream
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
) {
    int total_elements = N * C * H_out * W_out;
    // Choose block size and grid size
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to a reasonable number and use grid-stride loop
    if (blocks > 65535) blocks = 65535;
    avgpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding
    );
}

// Launch pow_scalar_forward
// Parameters:
//   input: device pointer to input tensor (contiguous float32)
//   output: device pointer to output tensor (same shape as input, allocated by caller)
//   exponent_ptr: device pointer to a single float value (0-dim tensor)
//   total_elements: number of elements
//   stream: CUDA stream
extern "C" void pow_scalar_forward_launcher(
    const float* input,
    float* output,
    const float* exponent_ptr,
    int total_elements,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    pow_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, exponent_ptr, total_elements
    );
}
