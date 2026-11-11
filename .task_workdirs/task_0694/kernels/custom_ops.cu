// custom_ops.cu
#include <cuda_runtime.h>
#include <cstdint>

// Define a helper for sigmoid
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused sum along dimension 1 and sigmoid kernel.
// Input: x with shape [N, C, H, W] and strides sxN, sxC, sxH, sxW.
// Output: y with shape [N, H, W] and strides syN, syH, syW, contiguous (syN=H*W, syH=W, syW=1).
// We assume num_axes is 1 (the second dimension) as per contract, but handle general dim for flexibility?
// To keep simple, we assume dim is the axis to reduce, and we reduce over that axis.
// Since C is small (3), each thread computes one output element by looping over C.
__global__ void sum_and_sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    long long input_stride_N, long long input_stride_C, long long input_stride_H, long long input_stride_W,
    long long output_stride_N, long long output_stride_H, long long output_stride_W,
    int64_t total_output_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_output_elements; i += stride) {
        // Convert linear index to (n, h, w) coordinates for output (dim reduction removed)
        int64_t n = i / ((int64_t)H * W);
        int64_t rem = i - n * ((int64_t)H * W);
        int h = (int)(rem / W);
        int w = (int)(rem - (int64_t)h * W);

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            // Input offset using original strides
            const float* in_ptr = input + n * input_stride_N + c * input_stride_C + h * input_stride_H + w * input_stride_W;
            sum += *in_ptr;
        }
        // Compute output offset (assuming output is contiguous: syN=H*W, syH=W, syW=1)
        float* out_ptr = output + n * output_stride_N + h * output_stride_H + w * output_stride_W;
        *out_ptr = sigmoidf(sum);
    }
}

// Separate sum kernel (no sigmoid) for debug
__global__ void sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    long long input_stride_N, long long input_stride_C, long long input_stride_H, long long input_stride_W,
    long long output_stride_N, long long output_stride_H, long long output_stride_W,
    int64_t total_output_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_output_elements; i += stride) {
        int64_t n = i / ((int64_t)H * W);
        int64_t rem = i - n * ((int64_t)H * W);
        int h = (int)(rem / W);
        int w = (int)(rem - (int64_t)h * W);

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float* in_ptr = input + n * input_stride_N + c * input_stride_C + h * input_stride_H + w * input_stride_W;
            sum += *in_ptr;
        }
        float* out_ptr = output + n * output_stride_N + h * output_stride_H + w * output_stride_W;
        *out_ptr = sum;
    }
}

// Elementwise sigmoid kernel
__global__ void sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = sigmoidf(input[i]);
    }
}

// Launcher for fused sum_and_sigmoid
extern "C" void custom_sum_and_sigmoid_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    long long input_stride_N, long long input_stride_C, long long input_stride_H, long long input_stride_W,
    long long output_stride_N, long long output_stride_H, long long output_stride_W,
    int64_t total_output_elements,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t blocks_needed = (total_output_elements + block_size - 1) / block_size;
    int grid_size = (int)min((int64_t)blocks_needed, (int64_t)65535); // grid max for x dimension; adjust if needed
    if (total_output_elements > (int64_t)65535 * block_size) {
        // Use grid-stride loop with 512 blocks (could be larger if needed, but 512 is fine)
        grid_size = 512;
    }
    sum_and_sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W,
        input_stride_N, input_stride_C, input_stride_H, input_stride_W,
        output_stride_N, output_stride_H, output_stride_W,
        total_output_elements);
}

// Launcher for sum only
extern "C" void custom_sum_only_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    long long input_stride_N, long long input_stride_C, long long input_stride_H, long long input_stride_W,
    long long output_stride_N, long long output_stride_H, long long output_stride_W,
    int64_t total_output_elements,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t blocks_needed = (total_output_elements + block_size - 1) / block_size;
    int grid_size = (int)min((int64_t)blocks_needed, (int64_t)65535);
    if (total_output_elements > (int64_t)65535 * block_size) {
        grid_size = 512;
    }
    sum_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W,
        input_stride_N, input_stride_C, input_stride_H, input_stride_W,
        output_stride_N, output_stride_H, output_stride_W,
        total_output_elements);
}

// Launcher for sigmoid only
extern "C" void custom_sigmoid_only_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t blocks_needed = (total_elements + block_size - 1) / block_size;
    int grid_size = (int)min((int64_t)blocks_needed, (int64_t)65535);
    if (total_elements > (int64_t)65535 * block_size) {
        grid_size = 512;
    }
    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(input, output, total_elements);
}
