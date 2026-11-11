// custom_ops.cu
// CUDA kernels for non-convolution ops only.
// Conv/ConvTranspose are implemented with cuDNN in C++ binding.

#include <cuda_runtime.h>

// Elementwise add with broadcast: output = input + bias
// input shape: [batch, channels, length]
// bias shape: [channels, 1]  (contiguous)
// output has same layout as input, so we assume output has same strides as input (passed in).
// We handle arbitrary strides by using 3D indexing.
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,   // stride for batch dimension (in elements)
    long long in_stride_channel, // stride for channel dimension
    long long in_stride_length,  // stride for length dimension
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * length;
    if (idx >= total) return;

    // Decompose flat index into (b, c, l)
    int l = idx % length;
    int tmp = idx / length;
    int c = tmp % channels;
    int b = tmp / channels;

    long long in_offset = b * in_stride_batch + c * in_stride_channel + l * in_stride_length;
    long long out_offset = b * out_stride_batch + c * out_stride_channel + l * out_stride_length;

    output[out_offset] = input[in_offset] + bias[c];
}

// Launcher for add_bias
// Input and output tensors are assumed to be on CUDA, with float32.
// input and output are passed as raw pointers; output already allocated.
// bias is contiguous float32 pointer of size channels.
// We also pass strides in elements (not bytes) for both input and output.
void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length,
    cudaStream_t stream
) {
    int total = batch * channels * length;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid excessive blocks, use grid-stride loop if needed.
    // For simplicity, we use exact number of blocks because total ~ 128*16*20480 = 41,943,040
    // which is within grid limit (2^31-1). However, to be safe, we could use grid-stride loop.
    // We'll implement grid-stride with maximum grid dimension (65535 for x).
    int max_grid_x = 65535;
    if (grid_size > max_grid_x) {
        // Use a grid-stride loop kernel variant (not implemented here) OR use multiple blocks
        // Since total is ~42M, grid_size ~ 164k > 65535, we need to handle large sizes.
        // We'll implement a simple grid-stride loop in the kernel by launching max_grid_x blocks.
        // Modify kernel to support grid-stride.
    }
    // Actually, we can launch with grid_size = min(grid_size, max_grid_x) and use grid-stride loop.
    // We'll create a separate kernel with grid-stride loop or modify existing kernel.
    // To keep it simple, we'll implement a grid-stride version inside add_bias_kernel by adding a stride parameter.
    // But we already have a simple kernel, so we can compute grid_x = min(grid_size, max_grid_x) and launch,
    // and kernel will process all elements using a loop based on total and grid size.
    // However, the kernel above doesn't have grid-stride loop; we need to add it.
    // Let's modify kernel to have grid-stride loop for robustness.
    // We'll rewrite add_bias_kernel with a grid-stride loop.
}

// Revised launcher and kernel with grid-stride loop:
__global__ void add_bias_kernel_gs(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total,
    int channels,
    int length,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int l = i % length;
        int tmp = i / length;
        int c = tmp % channels;
        int b = tmp / channels;

        long long in_offset = b * in_stride_batch + c * in_stride_channel + l * in_stride_length;
        long long out_offset = b * out_stride_batch + c * out_stride_channel + l * out_stride_length;
        output[out_offset] = input[in_offset] + bias[c];
    }
}

void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length,
    cudaStream_t stream
) {
    int total = batch * channels * length;
    int block_size = 256;
    int max_grid_x = 65535;
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > max_grid_x) {
        grid_size = max_grid_x;
    }
    add_bias_kernel_gs<<<grid_size, block_size, 0, stream>>>(
        input, bias, output, total, channels, length,
        in_stride_batch, in_stride_channel, in_stride_length,
        out_stride_batch, out_stride_channel, out_stride_length
    );
}

// C-style declaration for use by binding.cpp
extern "C" {
    void add_bias_launcher(
        const float* input,
        const float* bias,
        float* output,
        int batch,
        int channels,
        int length,
        long long in_stride_batch,
        long long in_stride_channel,
        long long in_stride_length,
        long long out_stride_batch,
        long long out_stride_channel,
        long long out_stride_length,
        cudaStream_t stream
    );
}
