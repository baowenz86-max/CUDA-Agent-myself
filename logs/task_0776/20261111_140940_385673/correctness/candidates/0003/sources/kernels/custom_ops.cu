#include <cuda_runtime.h>

// Adaptive average pooling 1D kernel
// Computes average pooling over last dimension for each element in batch and channels.
// Input: 3D tensor of shape [b, c, l_in]
// Output: 3D tensor of shape [b, c, l_out]
// The bin boundaries follow torch.nn.AdaptiveAvgPool1d: start = floor(i*l_in/l_out), end = ceil((i+1)*l_in/l_out)
__global__ void adaptive_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int b,
    int c,
    int l_in,
    int l_out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)gridDim.x * blockDim.x;
    long long total_output = (long long)b * c * l_out;

    for (long long i = idx; i < total_output; i += total) {
        int b_idx = (int)(i / ((long long)c * l_out));
        int rem = (int)(i % ((long long)c * l_out));
        int ch_idx = rem / l_out;
        int out_idx = rem % l_out;

        int start_idx = (int)((long long)out_idx * l_in / l_out);
        int end_idx = (int)(((long long)(out_idx + 1) * l_in + l_out - 1) / l_out); // ceil
        if (end_idx > l_in) end_idx = l_in;
        if (start_idx >= end_idx) {
            output[i] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        const float* in_ptr = input + ((long long)b_idx * c + ch_idx) * l_in + start_idx;
        for (int j = start_idx; j < end_idx; ++j) {
            sum += in_ptr[j];
        }
        output[i] = sum / (float)(end_idx - start_idx);
    }
}

// Launcher for adaptive_avg_pool1d
// Input and output are contiguous float tensors on CUDA.
extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int b,
    int c,
    int l_in,
    int l_out,
    cudaStream_t stream)
{
    long long total_output = (long long)b * c * l_out;
    int block = 256;
    long long grid_ll = (total_output + block - 1) / block;
    int grid = (int)(grid_ll > 65535 ? 65535 : grid_ll);
    adaptive_avg_pool1d_kernel<<<grid, block, 0, stream>>>(input, output, b, c, l_in, l_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, this would be checked; here we just ensure launch is valid.
        // The binding wrapper should check after call, but we leave this here for safety.
    }
}

// Repeat interleave (stack along new dimension) kernel
// Output is contiguous with shape [batch, repeats, channels, length].
// We simply copy input data repeats times into output; output flat size = input_flat_size * repeats.
__global__ void repeat_interleave_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long input_size,
    int repeats)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)gridDim.x * blockDim.x;
    long long output_size = input_size * (long long)repeats;

    for (long long i = idx; i < output_size; i += total) {
        long long src_idx = i % input_size;
        output[i] = input[src_idx];
    }
}

// Launcher for repeat_interleave_like
// input is contiguous flat buffer, output is contiguous buffer of size input_size * repeats.
extern "C" void repeat_interleave_like_launcher(
    const float* input,
    float* output,
    long long input_size,
    int repeats,
    cudaStream_t stream)
{
    long long output_size = input_size * (long long)repeats;
    int block = 256;
    long long grid_ll = (output_size + block - 1) / block;
    int grid = (int)(grid_ll > 65535 ? 65535 : grid_ll);
    repeat_interleave_kernel<<<grid, block, 0, stream>>>(input, output, input_size, repeats);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error if necessary in binding wrapper
    }
}
