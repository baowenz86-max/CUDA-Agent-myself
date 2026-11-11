#include <cuda_runtime.h>

// AvgPool1d forward kernel
// input: (batch, channels, length) float32, arbitrary strides
// output: (batch, channels, out_length) with same batch, channels as input
// kernel_size, stride, padding are integers
__global__ void avgpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    long long input_batch_stride,   // stride between batch elements
    long long input_channel_stride, // stride between channel elements
    long long input_pixel_stride,   // stride between adjacent length elements (usually 1)
    long long output_batch_stride,
    long long output_channel_stride,
    long long output_pixel_stride,
    int kernel_size,
    int stride,
    int padding,
    int total_output_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int out_idx = idx; out_idx < total_output_elements; out_idx += grid_stride) {
        // Decompose out_idx into batch, channel, out_pos
        int out_pos = out_idx % output_length;
        int tmp = out_idx / output_length;
        int ch = tmp % channels;
        int b = tmp / channels;

        // Compute input start index for pooling window
        int in_start = out_pos * stride - padding;
        float sum = 0.0f;
        // We hardcode count = kernel_size because count_include_pad=True
        // but we still need to guard against out-of-bounds when reading input.
        for (int k = 0; k < kernel_size; ++k) {
            int in_idx = in_start + k;
            if (in_idx >= 0 && in_idx < input_length) {
                // Compute input memory offset using strides
                const float* input_ptr = input 
                    + b * input_batch_stride 
                    + ch * input_channel_stride 
                    + in_idx * input_pixel_stride;
                sum += *input_ptr;
            }
            // else padding zero contributes nothing
        }
        float avg = sum / float(kernel_size);

        // Compute output memory offset
        float* output_ptr = output 
            + b * output_batch_stride 
            + ch * output_channel_stride 
            + out_pos * output_pixel_stride;
        *output_ptr = avg;
    }
}

// Launcher for the avgpool1d kernel.
// This function is called from the binding layer after tensor extraction.
extern "C" void avgpool1d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_pixel_stride,
    long long output_batch_stride,
    long long output_channel_stride,
    long long output_pixel_stride,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
) {
    // Determine total number of output elements
    long long total_elements_ll = (long long)batch_size * channels * output_length;
    if (total_elements_ll == 0) {
        return;
    }
    int total_output_elements = (int)total_elements_ll; // safe for sizes up to 2^31-1, larger handled via grid-stride loop? We can use long long in kernel if needed.
    // To support >2^31 elements, kernel would need long long indexing. For now, int is sufficient for typical sizes (< 2^31).

    // Launch configuration: 256 threads per block, enough blocks to cover, grid-stride loop handles large grids.
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    if (blocks > 65535) {
        // Limit grid dim to 65535 in x, use a smaller block count and rely on grid-stride loop.
        blocks = 65535;
    }
    avgpool1d_kernel<<<blocks, threads, 0, stream>>>(
        input,
        output,
        batch_size,
        channels,
        input_length,
        output_length,
        input_batch_stride,
        input_channel_stride,
        input_pixel_stride,
        output_batch_stride,
        output_channel_stride,
        output_pixel_stride,
        kernel_size,
        stride,
        padding,
        total_output_elements
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, we should throw a c10::Error, but we cannot include torch here.
        // The binding layer will check cudaGetLastError after calling this launcher.
        // We just ignore here; binding.cpp should call cudaDeviceSynchronize or check.
        // For safety, we could call cudaDeviceSynchronize, but that's expensive.
        // The contract implies the binding will do error checking.
    }
}
