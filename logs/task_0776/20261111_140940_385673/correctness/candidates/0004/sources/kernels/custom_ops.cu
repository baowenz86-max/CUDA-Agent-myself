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
    long long total_output = (long long)b * c * l_out;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_output;
         idx += stride)
    {
        int b_idx = (int)(idx / ((long long)c * l_out));
        int rem = (int)(idx % ((long long)c * l_out));
        int ch = rem / l_out;
        int out_idx = rem % l_out;

        int start_idx = (int)((long long)out_idx * l_in / l_out);
        int end_idx = (int)(((long long)(out_idx + 1) * l_in + l_out - 1) / l_out); // ceil
        if (end_idx > l_in) end_idx = l_in;
        if (start_idx >= end_idx) {
            output[idx] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        const float* in_ptr = input + (long long)b_idx * c * l_in + (long long)ch * l_in + start_idx;
        for (int j = start_idx; j < end_idx; ++j) {
            sum += in_ptr[j];
        }
        output[idx] = sum / (float)(end_idx - start_idx);
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
    if (grid < 1) grid = 1;
    adaptive_avg_pool1d_kernel<<<grid, block, 0, stream>>>(input, output, b, c, l_in, l_out);
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
    long long output_size = input_size * repeats;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < output_size;
         i += stride)
    {
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
    long long output_size = input_size * repeats;
    int block = 256;
    long long grid_ll = (output_size + block - 1) / block;
    int grid = (int)(grid_ll > 65535 ? 65535 : grid_ll);
    if (grid < 1) grid = 1;
    repeat_interleave_kernel<<<grid, block, 0, stream>>>(input, output, input_size, repeats);
}
